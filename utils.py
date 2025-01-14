import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.distributions import Normal
import GPyOpt
from GPyOpt.methods import BayesianOptimization
from tqdm import tqdm
from base import TrainingConfig, BaseModel, CNNConfig, LSTMConfig
from models import *
from Bioprocess_Sim import *
from CSTR_Sim import *
from dataclasses import fields
np.random.seed(42)

@dataclass
class ConformityScore:
    """Store different types of conformity scores for regression"""
    absolute_residual: Optional[np.ndarray] = None
    signed_residual: Optional[np.ndarray] = None
    normalized_residual: Optional[np.ndarray] = None
    cumulative_residual: Optional[np.ndarray] = None

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

class EarlyStopping:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, test_loss, model):
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.config.patience:
                self.early_stop = True

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

class ModelTrainer:
    """Handles model training and evaluation"""
    def __init__(self, model: BaseModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

    def train(self, train_loader: DataLoader, test_loader: DataLoader, 
              criterion: nn.Module) -> Dict[str, List[float]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay = self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.config.factor, patience=self.config.patience, verbose=True)
        early_stopping = EarlyStopping(self.config)
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
                scheduler.step(test_loss)
            else:
                test_loss = self._validate_epoch(test_loader, criterion)
                scheduler.step(test_loss)
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config.num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
                avg_train_loss = train_loss / len(train_loader)
                avg_test_loss = test_loss / len(test_loader)
                avg_loss = (avg_train_loss + avg_test_loss) / 2
                
                early_stopping(avg_test_loss, self.model)
                if early_stopping.early_stop:
                    print("Early Stopping")
                    break
                early_stopping.load_best_model(self.model)
                
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
                plt.fill_between(range(len(train_pred)), train_pred_new[min_key][:, i], train_pred_new[max_key][:, i], color='blue', alpha=0.2, edgecolor = 'None', label='Train Uncertainty')
                plt.fill_between(range(offset, offset + len(test_pred)), test_pred_new[min_key][:, i], test_pred_new[max_key][:, i], color='red', alpha=0.2, edgecolor = 'None',label='Test Uncertainty')
            
            if train_var is not None:
                plt.fill_between(range(len(train_pred)), train_pred[:, i] - np.sqrt(train_var[:, i]), train_pred[:, i] + np.sqrt(train_var[:, i]), color='blue', alpha=0.2, edgecolor = 'None',label='Train Uncertainty')
            if test_var is not None:
                plt.fill_between(range(offset, offset + len(test_pred)), test_pred[:, i] - np.sqrt(test_var[:, i]), test_pred[:, i] + np.sqrt(test_var[:, i]), color='red', alpha=0.2, edgecolor = 'None', label='Test Uncertainty')
            
            plt.show()
            
    @staticmethod
    def plot_loss(history: Dict[float, np.ndarray]):
        """Plots the loss history for a model"""
        plt.figure(figsize=(10, 6))
        for key, loss in history.items():
            plt.plot(loss, label=key)
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
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
    def __init__(self, model_class: BaseModel, sim_config: SimulationConfig, 
                train_config: TrainingConfig, model_config: Union[CNNConfig, LSTMConfig],
                config_bounds: Dict[str, Union[Tuple[float, float], List[Tuple[float, float]]]], # Modified to handle lists of bounds
                simulator, converter: SimulationConverter, data_processor: DataProcessor,
                trainer_class: ModelTrainer, iters: int = 100, quantiles: Optional[List[float]] = None, 
                monte_carlo = None):
        
        self.model_class = model_class
        self.sim_config = sim_config
        self.config_bounds = config_bounds
        self.train_config = train_config
        self.model_config = model_config
        self.data_processor = data_processor
        self.simulator = simulator(self.sim_config)
        self.trainer_class = trainer_class
        self.converter = converter()
        self.iters = iters
        self.quantiles = quantiles
        self.monte_carlo = monte_carlo
        self.domain = self._create_domain()
        print(self.domain)


    def _create_domain(self):
        """Create domain specification for Bayesian optimization"""
        domain = []
        
        for name, bounds in self.config_bounds.items():
            print(name)
            print(bounds)
            if isinstance(bounds, tuple):  # Single parameter bounds
                # param_type = 'discrete' if isinstance(getattr(self.train_config, name, 0), int) or \
                #             isinstance(getattr(self.model_config, name, 0), int) else 'continuous'
                param_type = 'continuous' if name == 'learning_rate' or name == 'dropout' else 'discrete'
                if param_type == 'discrete':
                    domain.append({
                        'name': name,
                        'type': 'discrete',
                        'domain': list(range(int(bounds[0]), int(bounds[1]) + 1))  # +1 to include upper bound
                    })
                else:
                    domain.append({
                        'name': name,
                        'type': 'continuous',
                        'domain': bounds
                    })
            elif isinstance(bounds, list):  # List parameter bounds (for CNN)
                for i, bound in enumerate(bounds):
                    domain.append({
                        'name': f'{name}_{i}',
                        'type': 'discrete',
                        'domain': list(range(int(bound[0]), int(bound[1]) + 1))  # +1 to include upper bound
                    })
        
        return domain

    def objective_function (self, x):
        # Initialise the hyperparameters to be optimised
        # self.train_config.hidden_dim = int(x[:, 0])
        # self.train_config.num_layers = int(x[:, 1])
        # self.train_config.dropout = float(x[:, 2])
        # self.train_config.learning_rate = float(x[:, 3])
        # self.train_config.batch_size = int(x[:, 4])
        # self.train_config.num_epochs = int(x[:, 5])
        # self.train_config.time_step = int(x[:, 6])
        
        # # Run the simulation
        # simulation_results = self.simulator.run_multiple_simulations()
        # features, targets = self.converter.convert(simulation_results)
        current_idx = 0
        print(self.config_bounds)
        # Process each parameter according to its type
        for name, bounds in self.config_bounds.items():
            if isinstance(bounds, tuple):  # Single parameter
                value = x[:, current_idx]
                current_idx += 1
                
                # Determine which config object to update
                if hasattr(self.train_config, name):
                    config_obj = self.train_config
                else:
                    config_obj = self.model_config
                    
                # Set the value with appropriate type
                if isinstance(getattr(config_obj, name), int):
                    setattr(config_obj, name, int(value))
                else:
                    setattr(config_obj, name, float(value))
                    
            elif isinstance(bounds, list):  # List parameter (for CNN)
                num_values = len(bounds)
                values = [x[:, current_idx + i] for i in range(num_values)]
                current_idx += num_values
                
                # Convert to appropriate type (assuming int for CNN parameters)
                values = [int(v) for v in values]
                setattr(self.model_config, name, values)

        # Rest of your existing objective function code...
        simulation_results = self.simulator.run_multiple_simulations()
        features, targets = self.converter.convert(simulation_results)        

        data_processor = self.data_processor(self.train_config)
        (train_loader, test_loader, X_train, X_test,
         y_train, y_test) = data_processor.prepare_data(features, targets)
        
        
        # Depending on which model, initialise the model parameters and loss function
        if self.model_class == QuantileLSTM or self.model_class == QuantileCNN:
            self.model = self.model_class(config = self.model_config, input_dim = X_train.shape[2], 
                                    output_dim = y_train.shape[1], quantiles = self.quantiles)
            criterion = QuantileLoss(self.quantiles)
        elif self.model_class == NLL_LSTM:
            criterion = nn.GaussianNLLLoss()
            self.model = self.model_class(config = self.model_config, input_dim = X_train.shape[2], 
                                    output_dim = y_train.shape[1])
        else:
            criterion = nn.MSELoss()
            self.model = self.model_class(config = self.model_config, input_dim = X_train.shape[2], 
                                    output_dim = y_train.shape[1])
        
        # Train the model
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
                    best_params = self.decode_parameters(best_params)
                
        return best_params
    
    def decode_parameters(self, x: torch.Tensor):
        """
        Decode the values of x into corresponding parameters and save them as a dictionary.
        
        Parameters:
        - x: torch.Tensor, Encoded parameters.
        - config_bounds: dict, Dictionary of parameter names and their bounds (tuple or list).
        - train_config: Object, Training configuration object.
        - model_config: Object, Model configuration object.
        
        Returns:
        - decoded_params: dict, Decoded parameters as a dictionary.
        """
        current_idx = 0
        decoded_params = {}

        for name, bounds in self.config_bounds.items():
            if isinstance(bounds, tuple):  # Single parameter
                value = x[current_idx]
                current_idx += 1
                
                # Determine which config object the parameter belongs to
                if hasattr(self.train_config, name):
                    config_obj = self.train_config
                else:
                    config_obj = self.model_config
                
                # Decode value with appropriate type
                if isinstance(getattr(config_obj, name), int):
                    decoded_value = int(value)
                else:
                    decoded_value = float(value)
                
                decoded_params[name] = decoded_value
            
            elif isinstance(bounds, list):  # List parameter (e.g., for CNN)
                num_values = len(bounds)
                values = [x[current_idx + i] for i in range(num_values)]
                current_idx += num_values
                
                # Convert to appropriate type (assuming int for list values)
                decoded_params[name] = [int(v) for v in values]
        
        return decoded_params

class ConformalQuantiles:
    """Class for calculating conformal quantiles based off test set predictions"""
    def __init__(self, y_test: np.ndarray, test_pred: Dict[float, np.ndarray], confidence: float):
        self.y_test = y_test # (time_steps, features)
        print(y_test.shape)
        self.test_pred = test_pred # {quantile: (time_steps, features)}
        self.confidence = confidence
        keys = test_pred.keys()
        self.upper_bound = test_pred[max(keys)]
        self.lower_bound = test_pred[min(keys)]
        
    def calculate_intervals(self):
        quantile_regression_calibration_intervals = np.zeros((self.y_test.shape[0], 2))
        scores = np.zeros((self.y_test.shape[0], self.y_test.shape[1]))
        print(scores.shape)
        for i in range(self.y_test.shape[0]):
            y_true = self.y_test[i, :]
            lower_bound = self.lower_bound[i, :]
            upper_bound = self.upper_bound[i, :]
            scores[i, :] = np.maximum(y_true - lower_bound, upper_bound - y_true)
            
            in_interval = np.logical_and(y_true >= lower_bound, y_true <= upper_bound)
            print(in_interval)
            quantile_regression_calibration_intervals[i, :] = (np.percentile(lower_bound, 100 * (1 - self.confidence / 2)), np.percentile(upper_bound, 100 * (1 - self.confidence / 2)))
        print(f"scores: {scores}")
        return quantile_regression_calibration_intervals

    
    def rank_residuals(self):
        residuals = self.residuals()
        rank_residuals = np.argsort(residuals, axis=0)
        return rank_residuals
    
class ConformalRegressor:
    """
    Implementation of various conformal prediction methods for regression.
    Supports both split-conformal and cross-conformal approaches.
    """
    def __init__(self, base_model: nn.Module, inverse_transformer, alpha: float = 0.1):
        self.base_model = base_model
        self.inverse_transformer = inverse_transformer
        self.alpha = alpha
        self.conformity_scores = ConformityScore()
        self.calibration_scores = None
        
    def _compute_absolute_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute absolute residuals |y - ŷ|"""
        return np.abs(y_true - y_pred)
    
    def _compute_normalized_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    sigma: np.ndarray) -> np.ndarray:
        """Compute normalized residuals |y - ŷ|/σ"""
        return np.abs(y_true - y_pred) / sigma
    
    def _compute_cumulative_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute cumulative distribution residuals"""
        residuals = y_true - y_pred
        return np.array([np.mean(residuals <= r) for r in residuals])
    
    def _compute_quantile_scores(self, y_true: np.ndarray, 
                               quantile_preds: Dict[float, np.ndarray]) -> np.ndarray:
        """Compute conformity scores for quantile regression"""
        scores = []
        for q, pred in quantile_preds.items():
            residual = y_true - pred
            score = np.maximum(q * residual, (q - 1) * residual)
            scores.append(score)
        return np.mean(scores, axis=0)

    def fit_calibrate(self, X_calib: torch.Tensor, y_calib: np.ndarray, 
                     method: str = 'absolute') -> None:
        """
        Fit the conformal predictor using calibration data.
        
        Args:
            X_calib: Calibration features
            y_calib: True calibration targets
            method: Conformity score method ('absolute', 'normalized', 'cumulative', 'quantile')
        """
        with torch.no_grad():
            if hasattr(self.base_model, 'quantiles'):
                y_pred = self.base_model(X_calib)
                quantile_preds = self.base_model.transform.inverse_transform(y_pred)
                scores = self._compute_quantile_scores(y_calib, quantile_preds)
            else:
                y_pred = self.base_model(X_calib).cpu().numpy()
                
                if method == 'absolute':
                    scores = self._compute_absolute_residuals(y_calib, y_pred)
                elif method == 'normalized':
                    # Estimate σ using MAD
                    residuals = y_calib - y_pred
                    mad = np.median(np.abs(residuals - np.median(residuals)))
                    sigma = 1.4826 * mad  # Consistent estimator for Gaussian data
                    scores = self._compute_normalized_residuals(y_calib, y_pred, sigma)
                elif method == 'cumulative':
                    scores = self._compute_cumulative_residuals(y_calib, y_pred)
                else:
                    raise ValueError(f"Unknown method: {method}")

        # Store calibration scores
        self.calibration_scores = scores
        
    def predict(self, X: torch.Tensor, method: str = 'absolute') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction intervals using conformal prediction.
        
        Args:
            X: Input features
            method: Conformity score method
            
        Returns:
            Tuple containing lower and upper bounds of the prediction interval
        """
        if self.calibration_scores is None:
            raise ValueError("Must call fit_calibrate before predict")
            
        with torch.no_grad():
            y_pred = self.base_model(X).cpu().numpy()
            
        # Compute quantile of conformity scores
        q = np.quantile(self.calibration_scores, 1 - self.alpha)
        
        if method == 'absolute':
            lower = y_pred - q
            upper = y_pred + q
        elif method == 'normalized':
            # Estimate σ for test points using similar approach as calibration
            residuals = self.calibration_scores  # Using calibration residuals
            mad = np.median(np.abs(residuals - np.median(residuals)))
            sigma = 1.4826 * mad
            lower = y_pred - q * sigma
            upper = y_pred + q * sigma
        elif method == 'cumulative':
            # For cumulative method, use empirical quantiles
            lower = np.percentile(y_pred, self.alpha/2 * 100)
            upper = np.percentile(y_pred, (1 - self.alpha/2) * 100)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return lower, upper
    
    def predict_quantile(self, X: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute prediction intervals using quantile regression with conformal calibration.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary containing predicted quantiles and conformal intervals
        """
        if not hasattr(self.base_model, 'quantiles'):
            raise ValueError("Base model must be a quantile regression model")
            
        with torch.no_grad():
            y_pred = self.base_model(X)
            quantile_preds = self.inverse_transformer.inverse_transform(y_pred)
            
        # Compute conformally calibrated intervals
        q = np.quantile(self.calibration_scores, 1 - self.alpha)
        
        results = {
            'quantiles': quantile_preds,
            'conformal_lower': quantile_preds[min(self.base_model.quantiles)] - q,
            'conformal_upper': quantile_preds[max(self.base_model.quantiles)] + q
        }
        
        return results

def main():
    # Configurations
    CSTR_sim_config = SimulationConfig(n_simulations=10, T=101, tsim=500)
    Biop_sim_config = SimulationConfig(n_simulations=10, T=20, tsim=240)
    training_config = TrainingConfig(
        batch_size=5,
        num_epochs=200000,
        learning_rate=0.001,
        time_step=10,
        weight_decay=0.01,
        factor=0.9,
        patience=10,
        delta = 0.1,
    )
    LSTM_Config = LSTMConfig(
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
    )
    CNN_Config = CNNConfig(
        conv_channels = [16, 32],
        kernel_sizes = [5, 3],
        fc_dims = [101, 128],
        dropout = 0.1
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
    # model = StandardCNN(
    #     config=training_config,
    #     input_dim=X_train.shape[2],
    #     output_dim=y_train.shape[1],
    # )
    quantiles = [0.01, 0.5, 0.99]
    model = QuantileCNN(
        config=CNN_Config,
        input_dim=X_train.shape[2],
        output_dim=y_train.shape[1], quantiles = quantiles
    )

    # Train model
    # criterion = nn.MSELoss()
    criterion = QuantileLoss(quantiles)
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

    # Inverse transform predictions
    scaler = data_processor.target_scaler
    inverse_transformer = QuantileTransform(quantiles, scaler)
    train_pred = inverse_transformer.inverse_transform(train_pred)
    test_pred = inverse_transformer.inverse_transform(test_pred)
    # train_pred = data_processor.target_scaler.inverse_transform(train_pred)
    # test_pred = data_processor.target_scaler.inverse_transform(test_pred)
    # train_var = data_processor.target_scaler.inverse_transform(train_var)
    # test_var = data_processor.target_scaler.inverse_transform(test_var)
    y_train_orig = data_processor.target_scaler.inverse_transform(y_train)
    y_test_orig = data_processor.target_scaler.inverse_transform(y_test)
    
    # mc_predictor = MC_Prediction(model, training_config, num_samples=100)
    # train_pred, train_var = mc_predictor.predict(X_train.to(training_config.device))
    # test_pred, test_var = mc_predictor.predict(X_test.to(training_config.device))
    
    # train_pred = data_processor.target_scaler.inverse_transform(train_pred)
    # test_pred = data_processor.target_scaler.inverse_transform(test_pred)
    # train_var = data_processor.target_scaler.inverse_transform(train_var)
    # test_var = data_processor.target_scaler.inverse_transform(test_var)

    # Visualize results
    feature_names = ['c_x', 'c_n', 'c_q']
    # feature_names = ['temperature', 'concentration']
    visualizer = Visualizer()
    visualizer.plot_predictions(train_pred, test_pred, y_train_orig, y_test_orig, feature_names, Biop_sim_config.n_simulations)
    visualizer.plot_loss(history)
    # model_class = QuantileCNN
    # trainer_class = ModelTrainer
    
    # CNN_config_bounds = {
    #     # Training config bounds
    #     'batch_size': (5, 10),
    #     'num_epochs': (50, 200),
    #     'learning_rate': (0.001, 0.1),
    #     'time_step': (5, 10),
        
    #     # CNN specific bounds
    #     'conv_channels': [(16, 32), (32, 64)],  # bounds for each conv layer
    #     'kernel_sizes': [(3, 5), (3, 7)],       # bounds for each kernel size
    #     'fc_dims': [(64, 128), (128, 256)],     # bounds for each FC layer
    #     'dropout': (0.1, 0.9)
    # }
    
    # LSTM_config_bounds = {
    #     # Training config bounds
    #     'batch_size': (5, 10),
    #     'num_epochs': (50, 200),
    #     'learning_rate': (0.001, 0.1),
    #     'time_step': (5, 10),
        
    #     # LSTM specific bounds
    #     'hidden_dim': (32, 128),
    #     'num_layers': (1, 3),
    #     'dropout': (0.1, 0.9),
    # }
    
    
    # optimizer = ModelOptimisation(model_class, Biop_sim_config, training_config, CNN_Config,
    #                               config_bounds=CNN_config_bounds, simulator=BioProcessSimulator, converter=BioprocessConverter, 
    #                               data_processor=DataProcessor, trainer_class=trainer_class, quantiles=quantiles, monte_carlo=None)
    
    # best_params = optimizer.optimise()
    # print(best_params)
    
    conformal = ConformalQuantiles(y_test_orig, test_pred, 0.95)
    intervals = conformal.calculate_intervals()
if __name__ == "__main__":
    main()