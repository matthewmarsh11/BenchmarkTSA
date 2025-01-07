import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    batch_size: int
    num_epochs: int
    learning_rate: float
    time_step: int
    train_test_split: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SimulationConfig:
    """Configuration for simulation data collection"""
    n_simulations: int
    T: int
    tsim: int
    noise_percentage: float = 0.01

class DataProcessor:
    """Handles data processing and preparation"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def prepare_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(len(features) - self.config.time_step - 1):
            X.append(features[i:i + self.config.time_step])
            y.append(targets[i + self.config.time_step])
        return np.array(X), np.array(y)

    def prepare_data(self, features: np.ndarray, targets: np.ndarray) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training"""
        # Scale data
        scaled_features = self.feature_scaler.fit_transform(features)
        scaled_targets = self.target_scaler.fit_transform(targets)

        # Create sequences
        X, y = self.prepare_sequences(scaled_features, scaled_targets)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Split data
        split_idx = int(len(X_tensor) * self.config.train_test_split)
        X_train = X_tensor[:split_idx]
        X_test = X_tensor[split_idx:]
        y_train = y_tensor[:split_idx]
        y_test = y_tensor[split_idx:]

        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, test_loader, X_train, X_test, y_train, y_test

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models"""
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class StandardLSTM(BaseModel):
    """Standard LSTM implementation"""
    def __init__(self, config: TrainingConfig, input_dim: int, hidden_dim: int, 
                 num_layers: int, output_dim: int, dropout: float = 0.2):
        super().__init__(config)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), 
                        self.lstm.hidden_size).to(self.config.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), 
                        self.lstm.hidden_size).to(self.config.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        return self.fc(lstm_out[:, -1, :])

class QuantileLSTM(BaseModel):
    """LSTM with quantile regression capabilities"""
    def __init__(self, config: TrainingConfig, input_dim: int, hidden_dim: int, 
                 num_layers: int, output_dim: int, quantiles: List[float], dropout: float = 0.2):
        super().__init__(config)
        self.quantiles = quantiles
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, len(quantiles) * output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), 
                        self.lstm.hidden_size).to(self.config.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), 
                        self.lstm.hidden_size).to(self.config.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions.view(-1, len(self.quantiles), self.output_dim)

class ModelTrainer:
    """Handles model training and evaluation"""
    def __init__(self, model: BaseModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

    def train(self, train_loader: DataLoader, test_loader: DataLoader, 
              criterion: nn.Module) -> Dict[str, List[float]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        history = {'train_loss': [], 'test_loss': []}

        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validation
            self.model.eval()
            test_loss = self._validate_epoch(test_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config.num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        return history

    def _train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer) -> float:
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            predictions = self.model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def _validate_epoch(self, test_loader: DataLoader, criterion: nn.Module) -> float:
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                total_loss += loss.item()
        return total_loss / len(test_loader)


class SimulationConverter():
    """Converts the simulation data into features and targets to be used in the model"""
    @abstractmethod
    def convert(self, data) -> Tuple[np.array, np.array]:
        """Convert output of simulation and return features and targets"""
        pass

from collections import defaultdict

class CSTRConverter(SimulationConverter):
    def convert(self, data: List[Tuple[Dict, Dict, Dict]]) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the process simulation data into features and targets

        Args:
            data (List[Tuple[Dict, Dict, Dict]]): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        obs_states = [obs for obs, _, _ in data]
        disturbances = [dist for _, dist, _ in data]
        actions = [act for _, _, act in data]
        
        combined_features = obs_states + disturbances + actions
        targets = [{k: v for k, v in obs.items() if k in ['Ca', 'T']} for obs in obs_states]
        
        aggregated_data = defaultdict(list)
        aggregated_targets = defaultdict(list)
         
        for d in combined_features:
            for key, value in d.items():
                aggregated_data[key].append(value)
        
        for d in targets:
            for key, value in d.items():
                aggregated_targets[key].append(value)
        
        all_features = []
        all_targets = []
        for key, value_list in aggregated_data.items():
            for value in value_list:
                all_features.append(value)
        
        for key, value_list in aggregated_targets.items():
            for value in value_list:
                all_targets.append(value)
        
        features = np.column_stack(all_features)
        targets = np.column_stack(all_targets)

        return features, targets
    
class BioprocessConverter(SimulationConverter):
    def convert(self, data: List[Tuple[Dict, Dict, Dict]]) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the process simulation data into features and targets

        Args:
            data (List[Tuple[Dict, Dict, Dict]]): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        
        obs_states = [obs for obs, _, _ in data]
        actions = [act for _, act, _ in data]

        combined_features = obs_states + actions
        combined_targets = obs_states
        
        agg_data = defaultdict(list)
        agg_targets = defaultdict(list)
        
        for d in combined_features:
            for key, value in d.items():
                agg_data[key].append(value)
        for d in combined_targets:
            for key, value in d.items():
                agg_targets[key].append(value)
                
        all_features = []
        all_targets = []
        for key, value_list in agg_data.items():
            for value in value_list:
                all_features.append(value)
                
        for key, value_list in agg_targets.items():
            for value in value_list:
                all_targets.append(value)
                
        features = np.column_stack(all_features)
        targets = np.column_stack(all_targets)
        

        return features, targets
        
        

class Visualizer:
    """Handles visualization of results"""
    @staticmethod
    def plot_predictions(train_pred: np.ndarray, test_pred: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray,
                        feature_names: List[str]):
        for i, name in enumerate(feature_names):
            plt.figure(figsize=(10, 6))
            
            # Plot training data
            plt.plot(train_pred[:, i], label='Train Predictions', color='blue')
            plt.plot(y_train[:, i], label='Train Ground Truth', color='green')
            
            # Plot testing data
            offset = len(train_pred)
            plt.plot(range(offset, offset + len(test_pred)), 
                    test_pred[:, i], label='Test Predictions', color='red')
            plt.plot(range(offset, offset + len(y_test)), 
                    y_test[:, i], label='Test Ground Truth', color='orange')
            
            plt.title(f'{name} Prediction')
            plt.xlabel('Time Step')
            plt.ylabel(name)
            plt.legend()
            plt.tight_layout()
            plt.show()

from CSTR_Sim import *
from Bioprocess_Sim import *
def main():
    # Configurations
    sim_config = SimulationConfig(n_simulations=10, T=101, tsim=500)
    training_config = TrainingConfig(
        batch_size=5,
        num_epochs=100,
        learning_rate=0.01,
        time_step=5
    )

    # Initialize components
    T = 20
    tsim = 240
    num_simulations = 10
    # simulator = CSTRSimulator(T, tsim)
    simulator = BioProcessSimulator(T, tsim)
    # Get data
    simulation_results = simulator.run_multiple_simulations(num_simulations)
    converter = BioprocessConverter()
    features, targets = converter.convert(simulation_results)
    
    data_processor = DataProcessor(training_config)
    # Prepare data
    (train_loader, test_loader, X_train, X_test, 
     y_train, y_test) = data_processor.prepare_data(features, targets)

    # Initialize model (example with StandardLSTM)
    model = StandardLSTM(
        config=training_config,
        input_dim=X_train.shape[2],
        hidden_dim=64,
        num_layers=2,
        output_dim=y_train.shape[1],
        dropout=0.2
    )

    # Train model
    trainer = ModelTrainer(model, training_config)
    history = trainer.train(train_loader, test_loader, nn.MSELoss())

    # Make predictions
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train.to(training_config.device)).cpu().numpy()
        test_pred = model(X_test.to(training_config.device)).cpu().numpy()

    # Inverse transform predictions
    train_pred = data_processor.target_scaler.inverse_transform(train_pred)
    test_pred = data_processor.target_scaler.inverse_transform(test_pred)
    y_train_orig = data_processor.target_scaler.inverse_transform(y_train)
    y_test_orig = data_processor.target_scaler.inverse_transform(y_test)

    # Visualize results
    feature_names = ['Concentration', 'Temperature']  # Example names
    visualizer = Visualizer()
    visualizer.plot_predictions(train_pred, test_pred, y_train_orig, y_test_orig, feature_names)

if __name__ == "__main__":
    main()