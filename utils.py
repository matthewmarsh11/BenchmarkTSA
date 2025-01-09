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
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.distributions import Normal
import GPyOpt
from GPyOpt.methods import BayesianOptimization
from tqdm import tqdm

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    batch_size: int
    num_epochs: int
    learning_rate: float
    time_step: int
    num_layers: int
    hidden_dim: int
    dropout: float = 0.2
    train_test_split: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SimulationConfig:
    """Configuration for simulation data collection"""
    n_simulations: int
    T: int
    tsim: int
    noise_percentage: float = 0.01

from collections import defaultdict

class SimulationConverter():
    """Converts the simulation data into features and targets to be used in the model"""
    @abstractmethod
    def convert(self, data) -> Tuple[np.array, np.array]:
        """Convert output of simulation and return features and targets"""
        pass

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

class DataProcessor:
    """Handles data processing and preparation"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def prepare_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(len(features) - self.config.time_step):
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
    def __init__(self, config: TrainingConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        self.lstm = nn.LSTM(
            input_dim, self.config.hidden_dim, self.config.num_layers,
            batch_first=True, dropout=self.config.dropout
        )
        self.fc = nn.Linear(self.config.hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.config.num_layers, x.size(0), 
                        self.config.hidden_dim).to(self.config.device)
        c0 = torch.zeros(self.config.num_layers, x.size(0), 
                        self.config.hidden_dim).to(self.config.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        return self.fc(lstm_out[:, -1, :])

class QuantileLSTM(BaseModel):
    """LSTM with quantile regression capabilities"""
    def __init__(self, config: TrainingConfig, input_dim: int, output_dim: int, quantiles: List[float]):
        super().__init__(config)
        self.quantiles = quantiles
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            input_dim, self.config.hidden_dim, self.config.num_layers,
            batch_first=True, dropout=self.config.dropout
        )
        self.fc = nn.Linear(self.config.hidden_dim, self.output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.config.num_layers, x.size(0), 
                        self.config.hidden_dim).to(self.config.device)
        c0 = torch.zeros(self.config.num_layers, x.size(0), 
                        self.config.hidden_dim).to(self.config.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        predictions = self.fc(lstm_out[:, -1, :])
        
        return predictions.view(-1, self.output_dim // len(self.quantiles), len(self.quantiles))

class NLL_LSTM(BaseModel):
    """LSTM using NLL Likelihood Loss Function, for Gaussian Likelihood, 
       contains second FC layer for log variance"""
    def __init__(self, config: TrainingConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        self.lstm = nn.LSTM(
            input_dim, self.config.hidden_dim, self.config.num_layers,
            batch_first=True, dropout=self.config.dropout
        )
        self.fc_mean = nn.Linear(self.config.hidden_dim, output_dim) # Fully connected layer for mean prediction
        self.fc_logvar = nn.Linear(self.config.hidden_dim, output_dim) # Fully connected layer for log variance prediction
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.config.num_layers, x.size(0),
                        self.config.hidden_dim).to(self.config.device)
        c0 = torch.zeros(self.config.num_layers, x.size(0),
                        self.config.hidden_dim).to(self.config.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        pred = self.fc_mean(lstm_out[:, -1, :]) # Mean prediction from model
        logvar = self.fc_logvar(lstm_out[:, -1, :])
        var = torch.exp(logvar) # Variance is exponent of logvar

        return pred, var

class MC_LSTM(BaseModel):
    """LSTM using Monte Carlo Dropout for uncertainty estimation"""
    def __init__(self, config: TrainingConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            self.input_dim, self.config.hidden_dim, self.config.num_layers,
            batch_first=True, dropout=self.config.dropout
        )
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.fc = nn.Linear(self.config.hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.config.num_layers, x.size(0),
                        self.config.hidden_dim).to(self.config.device)
        c0 = torch.zeros(self.config.num_layers, x.size(0),
                        self.config.hidden_dim).to(self.config.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        dropout_out = self.dropout(lstm_out[:, -1, :])
        return self.fc(dropout_out)

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
            if isinstance(criterion, nn.GaussianNLLLoss):
                train_loss = self._NLL_train_epoch(train_loader, criterion, optimizer)
            else:
                train_loss = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validation
            self.model.eval()
            if isinstance(criterion, nn.GaussianNLLLoss):
                test_loss = self._NLL_validate_epoch(test_loader, criterion)
            else:
                test_loss = self._validate_epoch(test_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config.num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
                avg_train_loss = train_loss / len(train_loader)
                avg_test_loss = test_loss / len(test_loader)
                avg_loss = (avg_train_loss + avg_test_loss) / 2
                
        return self.model, history, avg_loss

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
        return total_loss
    
    def _NLL_train_epoch(self, train_loader: DataLoader, criterion: nn.Module,
                        optimizer: torch.optim.Optimizer) -> float:
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            mean, var = self.model(batch_X)
            loss = criterion(mean, batch_y, var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _validate_epoch(self, test_loader: DataLoader, criterion: nn.Module) -> float:
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                total_loss += loss.item()
        return total_loss

    def _NLL_validate_epoch(self, test_loader: DataLoader, criterion: nn.Module) -> float:
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                mean, var = self.model(batch_X)
                loss = criterion(mean, batch_y, var)
                total_loss += loss.item()
        return total_loss

class Visualizer:
    """Handles visualization of results."""
    
    @staticmethod
    def plot_predictions(train_pred: Union[np.ndarray, Dict[float, np.ndarray]], 
                         test_pred: Union[np.ndarray, Dict[float, np.ndarray]],
                         y_train: np.ndarray, y_test: np.ndarray,
                         feature_names: list, num_simulations: int, train_var: Optional[np.ndarray] = None, test_var: Optional[np.ndarray] = None):
        """
        Plots predictions and ground truth data.
        
        Args:
            train_pred (np.ndarray): Training predictions (time steps, features).
            test_pred (np.ndarray): Testing predictions (time steps, features).
            y_train (np.ndarray): Training ground truth (time steps, features).
            y_test (np.ndarray): Testing ground truth (time steps, features).
            feature_names (list): List of feature names, one per column in the data.
            num_simulations (int): Number of simulations in the data.
        """
        
        feature_names = [f"{feature} Sim {i+1}" for feature in feature_names for i in range(num_simulations)]
        train_pred_new = None
        if isinstance(train_pred, dict):
            train_pred_new = train_pred
            test_pred_new = test_pred
            train_pred = train_pred[0.5]
            test_pred = test_pred[0.5]
            
        for i, sim in enumerate(feature_names):
            plt.figure(figsize=(10, 6))
            # Plot training predictions and ground truth
            plt.plot(train_pred[:, i], label=f'{sim} Train Predictions', color='blue', alpha=0.7)
            plt.plot(y_train[:, i], label=f'{sim} Train Ground Truth', color='green', alpha=0.7)
            
            # Plot testing predictions and ground truth
            offset = len(train_pred)
            plt.plot(range(offset, offset + len(test_pred)), 
                    test_pred[:, i], label=f'{sim} Test Predictions', color='red', alpha=0.7)
            plt.plot(range(offset, offset + len(y_test)), 
                    y_test[:, i], label=f'{sim} Test Ground Truth', color='orange', alpha=0.7)
        
            plt.title(f'{sim} Predictions')
            plt.xlabel('Time Step')
            plt.ylabel(sim)
            plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), ncol=1)
            plt.tight_layout()
            if train_pred_new:
                keys = train_pred_new.keys()
                max_key = max(keys)
                min_key = min(keys)
                plt.fill_between(range(len(train_pred)), train_pred_new[min_key][:, i], train_pred_new[max_key][:, i], color='blue', alpha=0.2, label='Train Uncertainty')
                plt.fill_between(range(offset, offset + len(test_pred)), test_pred_new[min_key][:, i], test_pred_new[max_key][:, i], color='red', alpha=0.2, label='Test Uncertainty')
            
            if train_var is not None:
                plt.fill_between(range(len(train_pred)), train_pred[:, i] - np.sqrt(train_var[:, i]), train_pred[:, i] + np.sqrt(train_var[:, i]), color='blue', alpha=0.2, label='Train Uncertainty')
            if test_var is not None:
                plt.fill_between(range(offset, offset + len(test_pred)), test_pred[:, i] - np.sqrt(test_var[:, i]), test_pred[:, i] + np.sqrt(test_var[:, i]), color='red', alpha=0.2, label='Test Uncertainty')
            
            plt.show()
   
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        Computes the quantile loss.

        Args:
            preds (torch.Tensor): Predicted quantiles of shape (batch_size, features * quantiles)
            target (torch.Tensor): Ground truth of shape (batch_size, features)

        Returns:
            torch.Tensor: The mean quantile loss.
        """
        # Reshape predictions to (batch_size, features, quantiles)
        num_features = target.size(1)
        preds = preds.view(-1, num_features, len(self.quantiles))

        assert not target.requires_grad
        assert preds.size(0) == target.size(0), "Batch size mismatch between preds and target"
        assert preds.size(1) == target.size(1), "Feature dimension mismatch between preds and target"

        # Initialize list to store losses for each quantile
        losses = []

        # Compute loss for each quantile
        for i, q in enumerate(self.quantiles):
            # Select the predictions for the i-th quantile
            pred_q = preds[:, :, i]  # Shape: (batch_size, features)

            # Compute the error (difference) between target and predicted quantile
            errors = target - pred_q

            # Quantile loss formula
            loss_q = torch.max((q - 1) * errors, q * errors)

            # Add the loss for this quantile to the list
            losses.append(loss_q.mean())

        # Mean loss across all quantiles
        total_loss = torch.stack(losses).mean()
        return total_loss

class QuantileTransform:
    def __init__(self, quantiles: List[float], scaler):
        
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.scaler = scaler
        
    def inverse_transform(self, preds):
        """
        Inverse transform the predictions from quantile space to original space.

        Args:
            preds (np.ndarray): Predictions of shape (batch_size, features * quantiles)

        Returns:
            np.ndarray: Predictions in original space of shape (batch_size, features)
        """

        # Initialize list to store inverse transformed predictions
        quantile_preds = {}
        # Inverse transform each quantile prediction
        for i, q in enumerate(self.quantiles):
            pred_q = preds[:, :, i]
            pred_q = self.scaler.inverse_transform(pred_q)
            quantile_preds[q] = pred_q # Shape: {quantile: (time_steps, features)}
        
        # Stack the predictions along the last axis
        return quantile_preds

class MC_Prediction:
    def __init__(self, model, config, num_samples):
        self.model = model
        self.config = config
        self.num_samples = num_samples

    def enable_dropout(self, m):
        if type(m) == nn.Dropout:
            m.train()
    
    def predict(self, data):
        self.model.eval()
        self.model.apply(self.enable_dropout)
        predictions = torch.zeros((self.num_samples, data.size(0), self.model.output_dim))
        print(predictions.shape)
        
        with torch.no_grad():
            for i in range(self.num_samples):
                predictions[i] = self.model(data)
                
        pred_mean = predictions.mean(dim=0)
        pred_var = predictions.var(dim=0)
        return pred_mean.numpy(), pred_var.numpy()

class ModelEvaluator:
    """
    Comprehensive model evaluation class supporting different types of predictions:
    - Point predictions (LSTM)
    - Probabilistic predictions with mean/std (NLL LSTM, MC Dropout)
    - Quantile predictions (Quantile LSTM)
    """
    
    def __init__(self, quantiles: Optional[List[float]] = None):
        """
        Initialize the model evaluator.
        
        Args:
            quantiles: List of quantiles for quantile-based models (e.g., [0.1, 0.5, 0.9])
        """
        self.quantiles = sorted(quantiles) if quantiles is not None else None
        
    def evaluate_deterministic(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate deterministic predictions (e.g., standard LSTM).
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        }
        return metrics
    
    def evaluate_probabilistic(self, y_true: np.ndarray, 
                             mean: np.ndarray, 
                             std: np.ndarray) -> Dict[str, float]:
        """
        Evaluate probabilistic predictions (e.g., NLL LSTM, MC Dropout).
        
        Args:
            y_true: Ground truth values
            mean: Predicted mean values
            std: Predicted standard deviation values
            
        Returns:
            Dictionary of metrics
        """
        # Calculate negative log likelihood
        dist = Normal(torch.Tensor(mean), torch.Tensor(std))
        nll = -dist.log_prob(torch.Tensor(y_true)).mean().item()
        
        # Calculate prediction intervals and coverage
        coverage_68 = np.mean(np.abs(y_true - mean) <= std)
        coverage_95 = np.mean(np.abs(y_true - mean) <= 2 * std)
        coverage_99 = np.mean(np.abs(y_true - mean) <= 3 * std)
        
        # Calculate calibration score (perfect calibration = 1.0)
        z_scores = (y_true - mean) / std
        _, p_value = stats.kstest(z_scores, 'norm')
        
        metrics = {
            'nll': nll,
            'coverage_68': coverage_68,
            'coverage_95': coverage_95,
            'coverage_99': coverage_99,
            'calibration_p_value': p_value,
            # Include deterministic metrics for the mean prediction
            **self.evaluate_deterministic(y_true, mean)
        }
        return metrics
    
    def evaluate_quantile(self, y_true: np.ndarray, 
                         predictions: Dict[float, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate quantile predictions (e.g., Quantile LSTM).
        
        Args:
            y_true: Ground truth values
            predictions: Dictionary mapping quantiles to predicted values
            
        Returns:
            Dictionary of metrics
        """
        if self.quantiles is None:
            raise ValueError("Quantiles must be specified for quantile evaluation")
            
        # Calculate quantile losses
        q_losses = {}
        for q, pred in predictions.items():
            errors = y_true - pred
            q_losses[f'quantile_loss_{q}'] = float(
                np.mean(np.maximum(q * errors, (q - 1) * errors))
            )
            
        # Calculate prediction interval coverage
        coverage = {}
        n = len(self.quantiles)
        for i in range(n//2):
            lower_q = self.quantiles[i]
            upper_q = self.quantiles[-(i+1)]
            interval_width = upper_q - lower_q
            
            lower_pred = predictions[lower_q]
            upper_pred = predictions[upper_q]
            
            in_interval = np.logical_and(y_true >= lower_pred, y_true <= upper_pred)
            coverage[f'coverage_{interval_width:.0%}'] = float(np.mean(in_interval))
            
        # Calculate CRPS
        crps = self._calculate_crps(y_true, predictions)
        
        # If median prediction available, include deterministic metrics
        median_metrics = {}
        if 0.5 in predictions:
            median_metrics = self.evaluate_deterministic(y_true, predictions[0.5])
            
        metrics = {
            'crps': crps,
            **q_losses,
            **coverage,
            **median_metrics
        }
        return metrics
    
    def evaluate_mc_dropout(self, y_true: np.ndarray, 
                          mc_samples: np.ndarray) -> Dict[str, float]:
        """
        Evaluate MC Dropout predictions.
        
        Args:
            y_true: Ground truth values
            mc_samples: Array of MC dropout samples (shape: n_samples x n_points)
            
        Returns:
            Dictionary of metrics
        """
        # Calculate mean and standard deviation across MC samples
        mean = np.mean(mc_samples, axis=0)
        std = np.std(mc_samples, axis=0)
        
        # Get probabilistic metrics
        prob_metrics = self.evaluate_probabilistic(y_true, mean, std)
        
        # Calculate additional MC Dropout specific metrics
        metrics = {
            **prob_metrics,
            'prediction_std_mean': float(np.mean(std)),
            'prediction_std_max': float(np.max(std)),
        }
        return metrics
    
    def _calculate_crps(self, y_true: np.ndarray, 
                       predictions: Dict[float, np.ndarray]) -> float:
        """Calculate Continuous Ranked Probability Score."""
        sorted_quantiles = sorted(predictions.keys())
        sorted_preds = np.stack([predictions[q] for q in sorted_quantiles], axis=-1)
        
        crps_sum = 0
        for i in range(len(sorted_quantiles)-1):
            q1, q2 = sorted_quantiles[i], sorted_quantiles[i+1]
            pred1, pred2 = sorted_preds[..., i], sorted_preds[..., i+1]
            
            width = q2 - q1
            height = (pred2 - pred1) / 2
            crps_sum += width * height
            
        return float(np.mean(crps_sum))
    
    def format_metrics(self, metrics: Dict[str, float], model_name: str) -> str:
        """
        Format metrics into a readable string.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model being evaluated
            
        Returns:
            Formatted string of metrics
        """
        lines = [f"\n{model_name} Evaluation Metrics:"]
        
        # Group metrics by type
        groups = {
            'Deterministic Metrics': ['mse', 'rmse', 'mae', 'mape', 'r2'],
            'Probabilistic Metrics': ['nll', 'calibration_p_value'],
            'Coverage Metrics': [k for k in metrics.keys() if 'coverage' in k],
            'Quantile Metrics': [k for k in metrics.keys() if 'quantile_loss' in k],
            'Other Metrics': ['crps', 'prediction_std_mean', 'prediction_std_max']
        }
        
        for group_name, metric_keys in groups.items():
            relevant_metrics = {k: metrics[k] for k in metric_keys if k in metrics}
            if relevant_metrics:
                lines.append(f"\n{group_name}:")
                for key, value in relevant_metrics.items():
                    if 'coverage' in key or 'mape' in key:
                        lines.append(f"  {key}: {value:.1%}")
                    else:
                        lines.append(f"  {key}: {value:.4f}")
                        
        return "\n".join(lines)

class ModelOptimisation:
    """Class for optimising parameters and hyperparameters of the model"""
    def __init__(self, model_class: BaseModel, sim_config: SimulationConfig, train_config: TrainingConfig, 
                 config_bounds: Dict[str, Tuple[float, float]],
                 simulator, converter: SimulationConverter, data_processor: DataProcessor,
                 trainer_class: ModelTrainer, iters: int = 100, quantiles: Optional[List[float]] = None, monte_carlo = None):
        self.model_class = model_class
        self.sim_config = sim_config
        self.config_bounds = config_bounds
        self.train_config = train_config
        self.data_processor = data_processor
        self.simulator = simulator(self.sim_config)
        self.trainer_class = trainer_class
        self.converter = converter()
        self.iters = iters
        self.quantiles = quantiles
        self.monte_carlo = monte_carlo
        print(self.config_bounds['hidden_dim'])
        self.domain = [{'name': 'hidden_dim', 'type': 'discrete', 'domain': range(self.config_bounds['hidden_dim'][0], self.config_bounds['hidden_dim'][1])},
                {'name': 'num_layers', 'type': 'discrete', 'domain': range(self.config_bounds['num_layers'][0], self.config_bounds['num_layers'][1])},
                {'name': 'dropout', 'type': 'continuous', 'domain': (self.config_bounds['dropout'][0], self.config_bounds['dropout'][1])},
                {'name': 'learning_rate', 'type': 'continuous', 'domain': (self.config_bounds['learning_rate'][0], self.config_bounds['learning_rate'][1])},
                {'name': 'batch_size', 'type': 'discrete', 'domain': range(self.config_bounds['batch_size'][0], self.config_bounds['batch_size'][1])},
                {'name': 'num_epochs', 'type': 'discrete', 'domain': range(self.config_bounds['num_epochs'][0], self.config_bounds['num_epochs'][1])},
                {'name': 'time_step', 'type': 'discrete', 'domain': range(self.config_bounds['time_step'][0], self.config_bounds['time_step'][1])}]

    
    def objective_function (self, x):
        # Initialise the hyperparameters to be optimised
        self.train_config.hidden_dim = int(x[:, 0])
        self.train_config.num_layers = int(x[:, 1])
        self.train_config.dropout = float(x[:, 2])
        self.train_config.learning_rate = float(x[:, 3])
        self.train_config.batch_size = int(x[:, 4])
        self.train_config.num_epochs = int(x[:, 5])
        self.train_config.time_step = int(x[:, 6])
        
        # Run the simulation
        simulation_results = self.simulator.run_multiple_simulations()
        features, targets = self.converter.convert(simulation_results)
        data_processor = self.data_processor(self.train_config)
        (train_loader, test_loader, X_train, X_test,
         y_train, y_test) = data_processor.prepare_data(features, targets)
        
        

        if isinstance(self.model_class, QuantileLSTM):
            self.model = self.model_class(config = self.train_config, input_dim = X_train.shape[2], 
                                    output_dim = y_train.shape[1], quantiles = self.quantiles)
            criterion = QuantileLoss(self.quantiles)
        elif isinstance(self.model_class, NLL_LSTM):
            criterion = nn.GaussianNLLLoss()
            self.model = self.model_class(config = self.train_config, input_dim = X_train.shape[2], 
                                    output_dim = y_train.shape[1])
        else:
            criterion = nn.MSELoss()
            self.model = self.model_class(config = self.train_config, input_dim = X_train.shape[2], 
                                    output_dim = y_train.shape[1])
        
        self.trainer = self.trainer_class(self.model, self.train_config)
        self.model, _, average_loss = self.trainer.train(train_loader, test_loader, criterion)
        
        if self.monte_carlo is not None:
            mc_predictor = self.monte_carlo(self.model, self.train_config, num_samples=100)
            _, train_var = mc_predictor.predict(X_train.to(self.train_config.device))
            average_loss = average_loss + np.concatenate(train_var).sum()
        
        return average_loss  
    
    
    def optimise(self):
        # Define the bounds for the hyperparameters
        optimizer = BayesianOptimization(
            f = self.objective_function,
            pbounds = self.config_bounds,
            domain = self.domain,
            model_type = 'GP',
            acquisition_type = 'EI',
            maximize = False
        )
        
        max_iter = self.iters
        with tqdm(total=max_iter, desc="Optimisation Progress") as pbar:
            best_loss = float('inf')
            for i in range(max_iter):
                optimizer.run_optimization(max_iter=1)
                pbar.update(1)
                pbar.set_postfix({'Loss': optimizer.fx_opt})
                if optimizer.fx_opt < best_loss:
                    best_loss = optimizer.fx_opt
                    best_params = optimizer.x_opt
                
        return best_params


    

from CSTR_Sim import *
from Bioprocess_Sim import *

def main():
    # Configurations
    CSTR_sim_config = SimulationConfig(n_simulations=10, T=500, tsim=101)
    Biop_sim_config = SimulationConfig(n_simulations=10, T=20, tsim=240)
    training_config = TrainingConfig(
        batch_size=5,
        num_epochs=100,
        learning_rate=0.01,
        time_step=5,
        num_layers=2,
        hidden_dim=64,
        dropout=0.2
    )

    # Initialize components

    # simulator = CSTRSimulator(CSTR_sim_config)
    simulator = BioProcessSimulator(Biop_sim_config)
    # Get data
    simulation_results = simulator.run_multiple_simulations()
    # converter = CSTRConverter()
    converter = BioprocessConverter()
    features, targets = converter.convert(simulation_results)
    
    data_processor = DataProcessor(training_config)
    # Prepare data
    (train_loader, test_loader, X_train, X_test, 
     y_train, y_test) = data_processor.prepare_data(features, targets)

    # Initialize model (example with StandardLSTM)
    # model = StandardLSTM(
    #     config=training_config,
    #     input_dim=X_train.shape[2],
    #     output_dim=y_train.shape[1],
    # )
    quantiles = [0.1, 0.5, 0.9]
    model = MC_LSTM(
        config=training_config,
        input_dim=X_train.shape[2],
        output_dim=y_train.shape[1]
    )

    # Train model
    criterion = nn.MSELoss()
    # criterion = QuantileLoss(quantiles)
    # criterion = nn.GaussianNLLLoss()
    trainer = ModelTrainer(model, training_config)
    model, history, avg_loss = trainer.train(train_loader, test_loader, criterion)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        if isinstance(criterion, nn.GaussianNLLLoss):
            train_pred, train_var = model(X_train.to(training_config.device))
            test_pred, test_var = model(X_test.to(training_config.device))
        else:
            train_pred = model(X_train.to(training_config.device)).cpu().numpy()
            test_pred = model(X_test.to(training_config.device)).cpu().numpy()
            print(train_pred.shape)

    # Inverse transform predictions
    scaler = data_processor.target_scaler
    # inverse_transformer = QuantileTransform(quantiles, scaler)

    # train_pred = inverse_transformer.inverse_transform(train_pred)
    # test_pred = inverse_transformer.inverse_transform(test_pred)
    # train_pred = data_processor.target_scaler.inverse_transform(train_pred)
    # test_pred = data_processor.target_scaler.inverse_transform(test_pred)
    # train_var = data_processor.target_scaler.inverse_transform(train_var)
    # test_var = data_processor.target_scaler.inverse_transform(test_var)
    y_train_orig = data_processor.target_scaler.inverse_transform(y_train)
    y_test_orig = data_processor.target_scaler.inverse_transform(y_test)
    
    mc_predictor = MC_Prediction(model, training_config, num_samples=100)
    train_pred, train_var = mc_predictor.predict(X_train.to(training_config.device))
    test_pred, test_var = mc_predictor.predict(X_test.to(training_config.device))
    
    train_pred = data_processor.target_scaler.inverse_transform(train_pred)
    test_pred = data_processor.target_scaler.inverse_transform(test_pred)
    train_var = data_processor.target_scaler.inverse_transform(train_var)
    test_var = data_processor.target_scaler.inverse_transform(test_var)

    # Visualize results
    feature_names = ['c_x', 'c_n', 'c_q']
    # visualizer = Visualizer()
    # visualizer.plot_predictions(train_pred, test_pred, y_train_orig, y_test_orig, feature_names, Biop_sim_config.n_simulations, train_var, test_var)
    model_class = MC_LSTM
    trainer_class = ModelTrainer
    optimizer = ModelOptimisation(model_class, Biop_sim_config, training_config, 
                                  config_bounds={'hidden_dim': (32, 128), 'num_layers': (1, 3), 
                                                 'dropout': (0.1, 0.5), 'learning_rate': (0.001, 0.1), 
                                                 'batch_size': (5, 10), 'num_epochs': (50, 200), 
                                                 'time_step': (5, 10)},
                                  simulator=BioProcessSimulator, converter=BioprocessConverter, 
                                  data_processor=DataProcessor, trainer_class=trainer_class, quantiles=None, monte_carlo=MC_Prediction)
    
    best_params = optimizer.optimise()
    print(best_params)
    
if __name__ == "__main__":
    main()