import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union, NamedTuple
from dataclasses import dataclass
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.distributions import Normal
import GPyOpt
from GPyOpt.methods import BayesianOptimization
from tqdm import tqdm
from base import *
from models import *
from Bioprocess_Sim import *
from CSTR_Sim import *
np.random.seed(42)
from fvcore.nn import FlopCountAnalysis

# Turn on Latex for last bit because it takes fucking ages

# plt.rcParams.update({
#     "text.usetex": True,  # Use LaTeX for text rendering
#     "font.family": "serif",  # Use a serif font
#     "font.serif": ["Computer Modern"],  # Default LaTeX font
#     "axes.labelsize": 14,  # Set font size for axes labels
#     "font.size": 12,  # Set global font size
#     "legend.fontsize": 12,  # Set legend font size
#     "xtick.labelsize": 12,  # Set x-axis tick size
#     "ytick.labelsize": 12,  # Set y-axis tick size
# })

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
        # The setpoint means nothing in this model as there is no defined 
        # relationship between the setpoint and the process

        for obs in obs_states:
            if 'Ca_s' in obs:
                del obs['Ca_s']
            
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

class ScalingResult(NamedTuple):
    """Container for scaled mean and variance results"""
    mean: np.ndarray
    variance: np.ndarray

class DataProcessor:
    """Handles data processing and preparation"""
    def __init__(self, config: TrainingConfig, features: np.ndarray, targets: np.ndarray):
        self.config = config
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.features = features
        self.targets = targets

    def prepare_sequences(self, scaled_features, scaled_targets) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(len(scaled_features) - self.config.time_step - self.config.horizon):
            # Input sequence remains the same
            X.append(scaled_features[i:i + self.config.time_step])
            
            # Get future values for all target variables
            future_values = scaled_targets[i + self.config.time_step:i + self.config.time_step + self.config.horizon]
            y.append(future_values)
        return np.array(X), np.array(y)

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training"""
        # Scale data
        scaled_features = self.feature_scaler.fit_transform(self.features)
        scaled_targets = self.target_scaler.fit_transform(self.targets)

        # Create sequences
        X, y = self.prepare_sequences(scaled_features, scaled_targets)
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Split data
        tt_idx = int(len(X_tensor) * self.config.train_test_split)
        tv_idx = int(len(X_tensor) * self.config.test_val_split)
        
        X_train = X_tensor[:tt_idx]
        X_test = X_tensor[tt_idx:tv_idx]
        X_val = X_tensor[tv_idx:]
        y_train = y_tensor[:tt_idx]
        y_test = y_tensor[tt_idx:tv_idx]
        y_val = y_tensor[tv_idx:]
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, test_loader, val_loader, X_train, X_test, X_val, y_train, y_test, y_val, X_tensor, y_tensor

    def rescale_predictions(self, scaled_mean: np.ndarray, scaled_variance: np.ndarray) -> ScalingResult:
        """
        Rescale the mean and variance predictions from the network back to the original scale.
        
        Parameters:
        -----------
        scaled_mean: array-like
            The mean predictions from the network (scaled between 0 and 1)
        scaled_variance: array-like
            The variance predictions from the network
            
        Returns:
        --------
        ScalingResult
            Named tuple containing rescaled mean and variance
        """
        # Get the scale factors from the target scaler
        
        if isinstance(scaled_mean, torch.Tensor):
            scaled_mean = scaled_mean.detach().numpy()
        if isinstance(scaled_variance, torch.Tensor):
            scaled_variance = scaled_variance.detach().numpy()
        
        scale_factor = self.target_scaler.data_max_ - self.target_scaler.data_min_

        
        rescaled_mean = self.target_scaler.inverse_transform(scaled_mean)
        rescaled_variance = np.multiply(scaled_variance, (scale_factor ** 2))

        return ScalingResult(rescaled_mean, rescaled_variance)
    
    def rescale_means(self, scaled_mean: np.ndarray) -> np.ndarray:
        
        if isinstance(scaled_mean, torch.Tensor):
            scaled_mean = scaled_mean.detach().numpy()
        
        rescaled_mean = np.zeros_like(scaled_mean)
        # Rescale the mean predictions
        for i in range(scaled_mean.shape[1]):
            rescaled_mean[:, i, :] = self.target_scaler.inverse_transform(scaled_mean[:, i, :])

        return rescaled_mean
    
    def reconstruct_sequence(self, sequence: np.ndarray, train_data: bool) -> np.ndarray:
        """
        Reconstruct the time series by averaging overlapping sequences.

        Parameters:
        -----------
        sequence: np.ndarray
            The sequence to reconstruct (shape: [num_sequences, time_horizon, num_features])
        train_data: bool
            Whether the sequence is training data or not
        Returns:
        --------
        np.ndarray
            The reconstructed time series (shape: [n_time_steps, num_features])
        """
        num_sequences, time_horizon, num_features = sequence.shape
        n_time_steps = self.targets.shape[0] - self.config.time_step - 1 # steps to predict - miss the last value because maths is pastied
        # if train_data:
        #     n_time_steps = int(num_sequences + self.config.train_test_split*(time_horizon + self.config.time_step))
        # else:
        #     n_time_steps = int(num_sequences + (1-self.config.train_test_split) * (time_horizon + self.config.time_step))
        # # Accumulators for sum and counts
        reconstructed = np.zeros((n_time_steps, num_features))
        count = np.zeros((n_time_steps, 1))

        for i in range(num_sequences):
            for h in range(time_horizon):
                t_index = i + h  # The actual time index in the full sequence
                reconstructed[t_index] += sequence[i, h]
                count[t_index] += 1

        # Avoid division by zero
        count[count == 0] = 1  

        return reconstructed / count

    def revert_sequences(self, train_mean: Union[np.ndarray, torch.Tensor], 
                            test_mean: Optional[Union[np.ndarray, torch.Tensor]] = None,
                            train_var: Optional[Union[np.ndarray, torch.Tensor]] = None,
                            test_var: Optional[Union[np.ndarray, torch.Tensor]] = None,
                            ) -> ScalingResult:
        """
        Process the model's output (mean and variance predictions) by rescaling them.
        
        Parameters:
        -----------
        train_mean: np.ndarray or torch.Tensor - the training mean output of the model
        train_var: Optional: np.ndarray or torch.Tensor - the test variance output of the model
        test_mean: Optional: np.ndarray or torch.Tensor - the test mean output of the model
        test_var: Optional: np.ndarray or torch.Tensor - the test variance output of the model
            
        Returns:
        --------
        ScalingResult
            Named tuple containing rescaled mean and variance
        """
        
        # Convert to numpy for scaling
        if isinstance(train_mean, torch.Tensor):
            train_mean = train_mean.detach().numpy()
        if isinstance(train_var, torch.Tensor):
            train_var = train_var.detach().numpy()
        if isinstance(test_mean, torch.Tensor):
            test_mean = test_mean.detach().numpy()
        if isinstance(test_var, torch.Tensor):
            test_var = test_var.detach().numpy()
        
        # Reconstruct the sequences into (time steps, observed preds, features)
        # The prediction only begins after the first sequence, so clip the last values
        train_mean = self.reconstruct_sequence(train_mean, True)
        if test_mean is not None:
            test_mean = self.reconstruct_sequence(test_mean, False)
# if its pastied remove last 10 again
        # Do it for the variance too
        if train_var is not None:
            train_var = self.reconstruct_sequence(train_var, True)

        if test_var is not None:
            test_var = self.reconstruct_sequence(test_var, False)


        if test_var is None:
            return self.target_scaler.inverse_transform(train_mean)
        
        means = np.concatenate([train_mean, test_mean], axis=0)
        if train_var is not None and test_var is not None:
            vars = np.concatenate([train_var, test_var], axis=0)
            return self.rescale_predictions(means, vars)
        else:
            return self.rescale_means(means)
        
        # if train_var is not None:
        #     return self.rescale_predictions(train_mean, train_var), self.rescale_predictions(test_mean, test_var)
        # else:
        #     return self.rescale_means(train_mean), self.rescale_means(test_mean)
        
    def quantile_invert(self, preds, quantiles):
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
        for i, q in enumerate(quantiles):
            pred_q = preds[:, :, :, i]
            # This is a prediction of shape (no_sequences, horizon, features)
            # Reconstruct the sequences
            pred_q = self.revert_sequences(pred_q)
            quantile_preds[q] = pred_q # Shape: {quantile: (time_steps, features)}
        
        # Stack the predictions along the last axis
        return quantile_preds
 
class EarlyStopping:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self.has_valid_state = False
    
    def __call__(self, test_loss, model):
        # Check if the loss is valid (not NaN)
        if not np.isnan(test_loss):
            # Only update best state if we have a valid loss that's better than previous best
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.best_model_state = model.state_dict()
                self.has_valid_state = True
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.config.patience:
                    self.early_stop = True
        else:
            # In case of NaN, increment counter but don't update model state
            self.counter += 1
            if self.counter >= self.config.patience:
                self.early_stop = True
    
    def load_best_model(self, model):
        if self.has_valid_state and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        else:
            # If we never got a valid state, keep the current model state
            pass
    
    def get_best_loss(self):
        return self.best_loss if self.has_valid_state else float('inf')

class ModelTrainer:
    """Handles model training and evaluation"""
    def __init__(self, model: BaseModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

    def train(self, train_loader: DataLoader, test_loader: DataLoader, val_loader: DataLoader,
            criterion: nn.Module) -> Dict[str, List[float]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay = self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.config.factor, patience=self.config.patience, verbose=True)
        early_stopping = EarlyStopping(self.config)
        history = {'train_loss': [], 'test_loss': [], 'val_loss': [], 'avg_loss': []}

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
                val_loss = self._NLL_validate_epoch(val_loader, criterion)
            else:
                test_loss = self._validate_epoch(test_loader, criterion)
                val_loss = self._validate_epoch(val_loader, criterion)
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_loss = (avg_train_loss + avg_test_loss) / 2
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['test_loss'].append(avg_test_loss)
            history['val_loss'].append(avg_val_loss)
            history['avg_loss'].append(avg_loss)
            
            # Use average loss for scheduler
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config.num_epochs}], '
                    f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, ' 
                    f'Val Loss: {avg_val_loss:.4f}'
                    f'Avg Loss: {avg_loss:.4f}')
                
                early_stopping(avg_loss, self.model)  # Use average loss for early stopping
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
    def plot_preds(preds: Union[np.ndarray, Dict[float, np.ndarray]],    
                         noisy_data, noiseless_data: np.ndarray, sequence_length: int,
                         time_horizon: int, feature_names: list, num_simulations: int, 
                         train_test_split: float, test_val_split: float,
                         vars: Optional[np.ndarray] = None):
        
        # If split the predictions into the train set and test set
        # If it is a dictionary (quantiles) need to account for this
        if isinstance(preds, np.ndarray):
            tt_idx = int(train_test_split * len(preds))
            tv_idx = int(test_val_split * len(preds))
        else:
            tt_idx = int(train_test_split * len(preds[0.5]))
            tv_idx = int(test_val_split * len(preds[0.5]))
                    
        pred_new = None
        
        feature_names = [f"{feature} Sim {i+1}" for feature in feature_names for i in range(num_simulations)]

        # Now have to deal with the case where the predictions are quantiles (dictionary)
        
        if isinstance(preds, dict):
            # Iterate through each quantile key and split the predictions
            tt_idx = int(train_test_split * len(preds[0.5]))
            tv_idx = int(test_val_split * len(preds[0.5]))

            pred_new = preds
            preds = preds[0.5]


        for i, sim in enumerate(feature_names):
            plt.figure(figsize=(10, 6))
            
            
            # Plot training predictions and ground truth
            plt.plot(range(sequence_length, sequence_length + len(preds[:tt_idx, i])),
                preds[:tt_idx, i], label=f'{sim} Train Predictions', color='blue', alpha=0.7)
            test_offset = sequence_length + len(preds[:tt_idx, i])
            plt.plot(range(test_offset, test_offset + len(preds[tt_idx:tv_idx, i])), 
                    preds[tt_idx:tv_idx, i], label=f'{sim} Test Predictions', color='red', alpha=0.7)
            val_offset = sequence_length + len(preds[:tt_idx, i]) + len(preds[tt_idx:tv_idx, i])
            plt.plot(range(val_offset, val_offset + len(preds[tv_idx:, i])), 
                    preds[tv_idx:, i], label=f'{sim} Val Predictions', color='cyan', alpha=0.7)
            
            plt.plot(noisy_data[:, i], label=f'{sim} Noisy Simulation', color='green', alpha=0.7)
            plt.plot(noiseless_data[:, i], label=f'{sim} Noiseless Data', color='black', linestyle = 'dashed', alpha=0.7)
            
            plt.title(f'{sim} Predictions')
            plt.xlabel('Time Step')
            plt.ylabel(sim)
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
            
            if vars is not None:
                plt.fill_between(range(sequence_length, sequence_length + len(preds[:tt_idx, i])),
                                preds[:tt_idx, i] - np.sqrt(vars[:tt_idx, i]), preds[:tt_idx, i] + np.sqrt(vars[:tt_idx, i]),
                                color='blue', alpha=0.2, edgecolor = 'None',label='Train Uncertainty')
                plt.fill_between(range(test_offset, test_offset + len(preds[tt_idx:tv_idx, i])),
                                preds[tt_idx:tv_idx, i] - np.sqrt(vars[tt_idx:tv_idx, i]), preds[tt_idx:tv_idx, i] + np.sqrt(vars[tt_idx:tv_idx, i]),
                                color='red', alpha=0.2, edgecolor = 'None', label='Test Uncertainty')
                plt.fill_between(range(val_offset, val_offset + len(preds[tv_idx:, i])),
                                preds[tv_idx:, i] - np.sqrt(vars[tv_idx:, i]), preds[tv_idx:, i] + np.sqrt(vars[tv_idx:, i]),
                                color='cyan', alpha=0.2, edgecolor = 'None', label='Val Uncertainty')
            # Plot the dictionary of quantiles as the uncertainty
            if pred_new:
                keys = pred_new.keys()
                max_key = max(keys)
                min_key = min(keys)
                plt.fill_between(range(sequence_length, sequence_length + len(preds[:tt_idx, i])),
                                pred_new[min_key][:tt_idx, i], pred_new[max_key][:tt_idx, i], color='blue', alpha=0.2, edgecolor = 'None', label='Train Uncertainty')
                plt.fill_between(range(test_offset, test_offset + len(preds[tt_idx:tv_idx, i])), 
                                pred_new[min_key][tt_idx:tv_idx, i], pred_new[max_key][tt_idx:tv_idx, i], color='red', alpha=0.2, edgecolor = 'None',label='Test Uncertainty')
                plt.fill_between(range(val_offset, val_offset + len(preds[tv_idx:, i])), 
                                pred_new[min_key][tv_idx:, i], pred_new[max_key][tv_idx:, i], color='cyan', alpha=0.2, edgecolor = 'None',label='Val Uncertainty')
            plt.tight_layout()
            plt.show()   
    
    @staticmethod
    def plot_predictions(preds: Union[np.ndarray, Dict[float, np.ndarray]],    
                         noisy_data, noiseless_data: np.ndarray, sequence_length: int,
                         time_horizon: int, feature_names: list, num_simulations: int, 
                         train_test_split: float,
                         vars: Optional[np.ndarray] = None):
        """
        Plots predictions and ground truth data.
        
        Args:
            train_pred (np.ndarray): Training predictions (time steps, features).
            test_pred (np.ndarray): Testing predictions (time steps, features).
            noisy_data (np.ndarray): Simulation with noise (time steps, features).
            noiseless_data (np.ndarray): Simulation without the noise (time steps, features).
            feature_names (list): List of feature names, one per column in the data.
            sequence_length (int): Length of the sequence.
            time_horizon (int): Time horizon for the predictions.
            num_simulations (int): Number of simulations in the data.
        """
        # Resplit back into train and test data.
        if isinstance(preds, np.ndarray):
            train_pred = preds[:int(train_test_split * len(preds))]
            test_pred = preds[int(train_test_split * len(preds)):]
        
        train_var = vars[:int(train_test_split * len(vars))] if vars is not None else None
        test_var = vars[int(train_test_split * len(vars)):] if vars is not None else None
                        
        feature_names = [f"{feature} Sim {i+1}" for feature in feature_names for i in range(num_simulations)]
        train_pred_new = None
        
        if isinstance(preds, dict):
            # Create empty dictionaries for training and testing predictions
            train_pred = {}
            test_pred = {}

            # Iterate through each quantile key and split the predictions
            split_idx = int(train_test_split * len(preds[0.5]))
            for key in preds.keys():
                train_pred[key] = preds[key][:split_idx]
                test_pred[key] = preds[key][split_idx:]
            train_pred_new = train_pred
            test_pred_new = test_pred
            train_pred = train_pred[0.5]
            test_pred = test_pred[0.5]
        
        for i, sim in enumerate(feature_names):
            plt.figure(figsize=(10, 6))
            
            # Plot training predictions and ground truth
            plt.plot(range(sequence_length, sequence_length + len(train_pred)),
                train_pred[:, i], label=f'{sim} Train Predictions', color='blue', alpha=0.7)
            
            
            # Plot testing predictions and ground truth
            offset = len(train_pred) + sequence_length + 2 - time_horizon
            test_offset = len(noisy_data) - len(test_pred)
            plt.plot(range(test_offset,  test_offset + len(test_pred)), 
                    test_pred[:, i], label=f'{sim} Test Predictions', color='red', alpha=0.7)
            
            plt.plot(noisy_data[:, i], label=f'{sim} Noisy Simulation', color='green', alpha=0.7)
            plt.plot(noiseless_data[:, i], label=f'{sim} Noiseless Data', color='black', linestyle = 'dashed', alpha=0.7)
            
            plt.title(f'{sim} Predictions')
            plt.xlabel('Time Step')
            plt.ylabel(sim)
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
            
            if train_pred_new:
                keys = train_pred_new.keys()
                max_key = max(keys)
                min_key = min(keys)
                plt.fill_between(range(sequence_length, sequence_length + len(train_pred)),
                                 train_pred_new[min_key][:, i], train_pred_new[max_key][:, i], color='blue', 
                                 alpha=0.2, edgecolor = 'None', label='Train Uncertainty')
                
                plt.fill_between(range(test_offset,  test_offset + len(test_pred)), 
                                 test_pred_new[min_key][:, i], test_pred_new[max_key][:, i], color='red', 
                                 alpha=0.2, edgecolor = 'None',label='Test Uncertainty')
            
            if train_var is not None:
                plt.fill_between(range(sequence_length, sequence_length + len(train_pred)), 
                                train_pred[:, i] - np.sqrt(train_var[:, i]), train_pred[:, i] + np.sqrt(train_var[:, i]),
                                color='blue', alpha=0.2, edgecolor = 'None',label='Train Uncertainty')
            if test_var is not None:
                plt.fill_between(range(test_offset,  test_offset + len(test_pred)),
                                 test_pred[:, i] - np.sqrt(test_var[:, i]), test_pred[:, i] + np.sqrt(test_var[:, i]),
                                 color='red', alpha=0.2, edgecolor = 'None', label='Test Uncertainty')
                
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def plot_actions(actions: np.ndarray, action_names: list, num_simulations: int):
        """Plots the actions of the simulation data"""
        action_names = [f"{action} Sim {i+1}" for action in action_names for i in range(num_simulations)]
        for i, action in enumerate(action_names):
            plt.figure(figsize=(10, 6))
            plt.plot(actions[:, i], label=action)
            plt.title(f'{action} action')
            plt.xlabel('Time Step')
            plt.ylabel(action)
            plt.legend()
            plt.show()
       
    @staticmethod
    def plot_conformal(train_pred: np.ndarray, test_pred: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
                       conformal_results: Dict, feature_names: list, num_simulations: int):
        feature_names = [f"{feature} Sim {i+1}" for feature in feature_names for i in range(num_simulations)]
        
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
            
            # plt.fill_between(range(len(train_pred)), conformal_results['conformal_intervals']['conformal_upper'][:, i], train_pred_new[max_key][:, i], color='blue', alpha=0.2, edgecolor = 'None', label='Train Uncertainty')
            plt.fill_between(range(offset, offset + len(test_pred)), conformal_results['conformal_intervals']['conformal_lower'][:, i], conformal_results['conformal_intervals']['conformal_upper'][:, i], color='red', alpha=0.2, edgecolor = 'None',label='Test Uncertainty')
            plt.tight_layout()
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
        

    @staticmethod
    def plot_loss_loss(history: Dict[str, List[float]]):
        """Plots the Pareto frontier of train vs test loss"""
        train_loss = np.array(history['train_loss'])
        test_loss = np.array(history['test_loss'])
        # Remove all losses greater than 1 for better visualization

        # Find Pareto frontier points
        pareto_points = []
        for i in range(len(train_loss)):
            dominated = False
            for j in range(len(train_loss)):
                if i != j:
                    if (train_loss[j] <= train_loss[i] and test_loss[j] <= test_loss[i] and 
                        (train_loss[j] < train_loss[i] or test_loss[j] < test_loss[i])):
                        dominated = True
                        break
            if not dominated:
                pareto_points.append((train_loss[i], test_loss[i]))
        
        pareto_points = np.array(sorted(pareto_points))
        
        plt.figure(figsize=(10, 6))
        # Plot all points
        plt.scatter(train_loss, test_loss, alpha=0.5, label='All points')
        # Plot Pareto frontier
        if len(pareto_points) > 0:
            plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'r-', label='Pareto frontier')
            plt.scatter(pareto_points[:, 0], pareto_points[:, 1], c='r', label='Pareto points')
        
        plt.xlabel('Training Loss')
        plt.ylabel('Test Loss')
        plt.title('Pareto Frontier of Training vs Test Loss')
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
        num_features = target.size(2)
        horizon = target.size(1)
        preds = preds.view(-1, horizon, num_features, len(self.quantiles))

        assert not target.requires_grad
        assert preds.size(0) == target.size(0), "Batch size mismatch between preds and target"
        assert preds.size(1) == target.size(1), "Feature dimension mismatch between preds and target"

        # Initialize list to store losses for each quantile
        losses = []

        # Compute loss for each quantile
        for i, q in enumerate(self.quantiles):
            # Select the predictions for the i-th quantile
            pred_q = preds[:, :, :, i]  # Shape: (batch_size, features)

            # Compute the error (difference) between target and predicted quantile
            errors = target - pred_q

            # Quantile loss formula
            loss_q = torch.max((q - 1) * errors, q * errors)

            # Add the loss for this quantile to the list
            losses.append(loss_q.mean())

        # Mean loss across all quantiles
        total_loss = torch.stack(losses).mean()
        return total_loss

class MC_Prediction:
    def __init__(self, model, num_samples):
        self.model = model
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

class ModelEvaluation:
    """Model evaluation class, to determine the prediction error (MSE, MAE etc.),
    also predicts the coverage interval, based on the prediction uncertainty"""
    def __init__(self, model: BaseModel, y_test: np.ndarray, y_pred: Union[np.ndarray, Dict[float, np.ndarray]],
                 test_var: Optional[np.ndarray] = None):
        self.model = model
        self.y_test = y_test
        self.y_pred = y_pred
        self.test_var = test_var if test_var is not None else None
        
    def MSE(self):
        if isinstance(self.y_pred, dict):
            y_pred = self.y_pred[0.5]
        else:
            y_pred = self.y_pred
        return np.mean((self.y_test - y_pred) ** 2)
    
    def MAE(self):
        if isinstance(self.y_pred, dict):
            y_pred = self.y_pred[0.5]
        else:
            y_pred = self.y_pred
        return np.mean(np.abs(self.y_test - y_pred))
    
    def RMSE(self):
        return np.sqrt(self.MSE())
    
    def MAPE(self):
        if isinstance(self.y_pred, dict):
            y_pred = self.y_pred[0.5]
        else:
            y_pred = self.y_pred
        return np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
    
    def coverage(self):
        """Calculate coverage ratios at 80%, 90%, 95% and 99% confidence intervals.
        
        Returns:
            dict: Coverage ratios for each confidence level
        """
        target_coverages = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        coverage_results = {}
        
        if isinstance(self.y_pred, dict):
            # Handle quantile-based predictions
            for target_coverage in target_coverages.keys():
                lower_quantile = (1 - target_coverage) / 2
                upper_quantile = 1 - lower_quantile
                
                # Get or interpolate quantile predictions
                if lower_quantile in self.y_pred and upper_quantile in self.y_pred:
                    lower_bound = self.y_pred[lower_quantile]
                    upper_bound = self.y_pred[upper_quantile]
                    coverage_ratio = np.mean((self.y_test >= lower_bound) & 
                                          (self.y_test <= upper_bound))
                    coverage_results[target_coverage] = coverage_ratio
        
        elif self.test_var is not None:
            # Handle variance-based predictions (NLL or MC Dropout)
            std_dev = np.sqrt(self.test_var)
            
            for target_coverage, z_score in target_coverages.items():
                lower_bound = self.y_pred - z_score * std_dev
                upper_bound = self.y_pred + z_score * std_dev
                coverage_ratio = np.mean((self.y_test >= lower_bound) & 
                                      (self.y_test <= upper_bound))
                coverage_results[target_coverage] = coverage_ratio
                
        return coverage_results

class ModelOptimisation:
    """Class for optimising parameters and hyperparameters of the model"""
    def __init__(self, model_class: BaseModel, train_config: TrainingConfig, model_config: Union[CNNConfig, LSTMConfig],
                config_bounds: Dict[str, Union[Tuple[float, float], List[Tuple[float, float]]]], 
                features_path: str, targets_path: str, converter: SimulationConverter, data_processor: DataProcessor,
                trainer_class: ModelTrainer, iters: int = 100, quantiles: Optional[List[float]] = None, 
                monte_carlo: Optional[classmethod] = None, variance: Optional[bool] = None):
        
        self.model_class = model_class
        self.config_bounds = config_bounds
        self.train_config = train_config
        self.model_config = model_config
        self.features_path = features_path
        self.targets_path = targets_path
        self.data_processor = data_processor
        self.trainer_class = trainer_class
        self.converter = converter()
        self.iters = iters
        self.quantiles = quantiles
        self.monte_carlo = monte_carlo
        self.variance = variance
        self.domain = self._create_domain()
        self.mdl_num = 0


    def _create_domain(self):
        """Create domain specification for Bayesian optimization"""
        domain = []
        
        for name, bounds in self.config_bounds.items():
            if isinstance(bounds, tuple):  # Single parameter bounds

                param_type = ('continuous' if name == 'learning_rate' or name == 'dropout' or
                              name == 'weight_decay' or name == 'factor' or name == 'delta'
                              else 'discrete'
                )
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

    def initialize_model_and_criterion(self):
        """
        Initialize model and criterion with support for multiple uncertainty estimation methods:
        - Quantile regression
        - Monte Carlo dropout
        - Variance estimation
        
        Returns:
            tuple: (model, criterion, use_monte_carlo)
        """
        if self.model_class == MLP or self.model_class == MLR:
            model_kwargs = {
                'config': self.model_config,
                'input_dim': self.X_train.shape[1] * self.X_train.shape[2],
                'output_dim': self.y_train.shape[2],
                'horizon': self.train_config.horizon
            }
        else:
            model_kwargs = {
                'config': self.model_config,
                'input_dim': self.X_train.shape[2],
                'output_dim': self.y_train.shape[2],
                'horizon': self.train_config.horizon
            }
            
        # Start with base configuration
        use_monte_carlo = False
        if hasattr(self.model_config, 'num_heads') and hasattr(self.model_config, 'd_model'):
            # Ensure d_model is divisible by num_heads
            if self.model_config.d_model % self.model_config.num_heads != 0:
                # Round d_model up to nearest multiple of num_heads
                self.model_config.d_model = ((self.model_config.d_model + self.model_config.num_heads - 1) // self.model_config.num_heads) * self.model_config.num_heads
        
        if self.variance:
            model_kwargs['var'] = True
            criterion = nn.GaussianNLLLoss()
        elif self.quantiles is not None:
            model_kwargs['quantiles'] = self.quantiles
            criterion = QuantileLoss(self.quantiles)
        else:
            criterion = nn.MSELoss()
        
        # Monte Carlo can be combined with other methods
        if self.monte_carlo:
            model_kwargs['monte_carlo'] = True
            use_monte_carlo = True
        
        # Initialize model with collected parameters
        self.model = self.model_class(**model_kwargs)
        
        return self.model, criterion, use_monte_carlo

    def objective_function (self, x):
        # Get the parameters in terms of x
        current_idx = 0
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
                # Map 0 to False and 1 to True for bidirectional LSTM
                if name == 'bidirectional':
                    setattr(config_obj, name, bool(int(value)))
                if name == 'norm_type':
                    norm_types = {0: None, 1: 'layer', 2: 'batch'}
                    setattr(config_obj, name, norm_types[int(value)])
                    
                if name == 'activation':
                    activations = {0: "ReLU", 1: "Softplus", 2: "Tanh", 3: "SELU", 4: "LeakyReLU", 5: "Sigmoid", 6: "Softmax", 7: "LogSoftmax"}
                    setattr(config_obj, name, activations[int(value)])
                
            elif isinstance(bounds, list):  # List parameter (for CNN)
                num_values = len(bounds)
                values = [x[:, current_idx + i] for i in range(num_values)]
                current_idx += num_values
                
                # Convert to appropriate type (assuming int for CNN parameters)
                values = [int(v) for v in values]
                setattr(self.model_config, name, values)

        features = pd.read_csv(self.features_path)
        targets = pd.read_csv(self.targets_path)      
        data_processor = self.data_processor(self.train_config, features, targets)

        (train_loader, test_loader, val_loader, self.X_train, self.X_test, self.X_val,
        self.y_train, self.y_test, self.y_val, self.X, self.y) = data_processor.prepare_data()
        
            
        self.model, criterion, use_monte_carlo = self.initialize_model_and_criterion()
        
        # Train the model
        self.trainer = self.trainer_class(self.model, self.train_config)
        self.model, _, average_loss = self.trainer.train(train_loader, test_loader, val_loader, criterion)
        
        if np.isnan(average_loss):
            average_loss = 1e6
        
        # Apply Monte Carlo prediction if enabled
        # Can you actually optimise to reduce aleatoric uncertainty?
        if use_monte_carlo:
            mc_predictor = self.monte_carlo(self.model, num_samples=100)
            _, train_var = mc_predictor.predict(self.X_train.to(self.train_config.device))
            average_loss = average_loss + np.concatenate(train_var).sum()
            
        return average_loss
    
    
    def optimise(self, path: str = None):
        # Define the bounds for the hyperparameters
        optimizer = BayesianOptimization(
            f = self.objective_function,
            pbounds = self.config_bounds,
            domain = self.domain,
            model_type = 'GP',
            acquisition_type = 'EI',
            maximize = False
        )
        
        if self.variance:
            max_iter = self.iters
            with tqdm(total=max_iter, desc="Optimisation Progress", position=0, leave=True) as pbar:
                best_loss = float('inf')
                for i in range(max_iter):
                    optimizer.run_optimization(max_iter=1)
                    pbar.update(1)
                    pbar.set_postfix({'Loss': optimizer.fx_opt})
                    if optimizer.fx_opt < -0.5:
                        best_loss = optimizer.fx_opt
                        best_params = optimizer.x_opt
                        best_params = self.decode_parameters(best_params)
                        
                        # Fix the path handling
                        base_path = path.rsplit('.', 1)[0]  # Remove the extension
                        if '_model_' in base_path:
                            base_path = base_path.split('_model_')[0]  # Remove any existing model number
                        new_path = f"{base_path}_model_{self.mdl_num}.pth"
                        
                        checkpoint = {
                            'model_state_dict': self.model.state_dict(),
                            'model_config': self.model_config,
                            'training_config': self.train_config,
                        }
                        
                        if self.model_class == MLP or self.model_class == MLR:
                            input_dim = self.X_train.shape[1] * self.X_train.shape[2]
                        else:
                            input_dim = self.X_train.shape[2]
                            
                        model = self.model_class(checkpoint['model_config'],
                                                input_dim=input_dim,
                                                output_dim=self.y_train.shape[2],  
                                                horizon=checkpoint['training_config'].horizon,
                                                var=True)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.eval()
                        
                        with torch.no_grad():
                            mean, var = model(self.X)
                            mse = nn.MSELoss()
                            self.mse_loss = mse(mean, self.y)
                            self.avg_var = torch.mean(var)
                            checkpoint['var'] = self.avg_var
                            checkpoint['mse'] = self.mse_loss
                
                            torch.save(checkpoint, new_path)
                            self.mdl_num += 1
                
            return best_params, best_loss
                        
        else:
            max_iter = self.iters
            with tqdm(total=max_iter, desc="Optimisation Progress", position=0, leave=True) as pbar:
                best_loss = float('inf')
                for i in range(max_iter):
                    optimizer.run_optimization(max_iter=1)
                    pbar.update(1)
                    pbar.set_postfix({'Loss': optimizer.fx_opt})
                    if optimizer.fx_opt < best_loss:
                        best_loss = optimizer.fx_opt
                        best_params = optimizer.x_opt
                        best_params = self.decode_parameters(best_params)
                        checkpoint = {
                            'model_state_dict': self.model.state_dict(),
                            'model_config': self.model_config,
                            'training_config': self.train_config,
                        }
                        torch.save(checkpoint, path)
                    
            return best_params, best_loss
    
    def save_model(self, model, path):
        if self.variance:
            torch.save({'model_state_dict': model.state_dict(),
                    'config': self.model_config,
                    'train_config': self.train_config,
                    'var': self.avg_var,
                    'mse': self.mse_loss
                    }, path)
        else:
            torch.save({'model_state_dict': model.state_dict(),
                    'config': self.model_config,
                    }, path)
    
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

class ConformalQuantile:
    """
    Implementation of various conformal prediction methods for regression.
    Supports both split-conformal and cross-conformal approaches.
    Uses the normalised values
    """
    def __init__(self, base_model: nn.Module, scaler, X_test: torch.Tensor, y_test: torch.Tensor, alpha: float = 0.1):
        self.base_model = base_model
        self.scaler = scaler
        self.X_test = X_test
        self.y_test = y_test
        self.alpha = alpha
        self.conformity_scores = ConformityScore()
        self.y_pred = self.base_model(X_test)
        self.calibration_scores = self.fit_calibrate()
        
    def _compute_absolute_residuals(self) -> np.ndarray:
        """Compute absolute residuals |y - ŷ|"""
        return np.abs(self.y_test - self.y_pred)
    
    def _compute_quantile_scores(self, quantile_preds: Dict[float, np.ndarray]) -> np.ndarray:
        """Compute conformity scores for quantile regression"""
        scores = []
        for q, pred in quantile_preds.items():
            residual = self.y_test - pred.numpy()
            score = np.maximum(q * residual, (q - 1) * residual)
            
            scores.append(score)
        return np.mean(scores, axis=0)

    def fit_calibrate(self) -> None:
        """
        Fit the conformal predictor using calibration data.
        
        Args:
            X_calib: Calibration features
            y_calib: True calibration targets
            method: Conformity score method ('absolute', 'normalized', 'cumulative', 'quantile')
        """
        with torch.no_grad():
            self.quantile_preds = {}
            for i, q in enumerate(self.base_model.quantiles):
                pred_q = self.y_pred[:, :, i]
                self.quantile_preds[q] = pred_q
            # Scores use the unnormalised quantile predictions
            scores = self._compute_quantile_scores(self.quantile_preds)
        return scores

    
    def predict_quantile(self) -> Dict[str, np.ndarray]:
        """
        Compute prediction intervals using quantile regression with conformal calibration.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary containing predicted quantiles and conformal intervals
        """

            
        # Get number of columns/features
        num_features = self.calibration_scores.shape[1]
        # Qs is the 90% quantile across each feature for every time step
        qs = np.array([np.quantile(self.calibration_scores[:, i], 1 - self.alpha) for i in range(num_features)])
        
        # Initialize arrays for lower and upper bounds
        conformal_lower = np.zeros_like(self.quantile_preds[min(self.base_model.quantiles)].detach().numpy())
        conformal_upper = np.zeros_like(self.quantile_preds[max(self.base_model.quantiles)].detach().numpy())
        
        # Apply feature-specific conformal scores
        for i in range(num_features):
            conformal_lower[:, i] = self.quantile_preds[0.5][:, i].detach().numpy() - qs[i]
            conformal_upper[:, i] = self.quantile_preds[0.5][:, i].detach().numpy() + qs[i]
        
        results = {
            'quantiles': self.quantile_preds,
            'conformal_lower': conformal_lower,
            'conformal_upper': conformal_upper
        }
        
        return results
    
    def _find_equivalent_quantile(self, conformal_interval: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Find the quantiles that correspond to the conformal prediction interval bounds.
        
        Args:
            self.y_test: True values
            conformal_interval: Dictionary containing conformal prediction bounds
            
        Returns:
            Tuple of (lower_quantile, upper_quantile) that match the conformal coverage
        """
        lower_bound = conformal_interval['conformal_lower']
        upper_bound = conformal_interval['conformal_upper']
        
        # Calculate the empirical coverage of the conformal interval
        in_interval = np.logical_and(self.y_test.numpy() >= lower_bound, self.y_test.numpy() <= upper_bound)
        coverage = np.mean(in_interval)
        
        def interpolate_quantile(q: float) -> np.ndarray:
            """
            Interpolate between available quantiles for a given q value.
            """
            # Get available quantiles and sort them
            quant_keys = sorted(list(conformal_interval['quantiles'].keys()))
            
            # Find the two closest quantiles
            idx = np.searchsorted(quant_keys, q)
            if idx == 0:
                return conformal_interval['quantiles'][quant_keys[0]]
            elif idx == len(quant_keys):
                return conformal_interval['quantiles'][quant_keys[-1]]
            
            # Interpolate between the two closest quantiles
            q1, q2 = quant_keys[idx-1], quant_keys[idx]
            v1 = conformal_interval['quantiles'][q1]
            v2 = conformal_interval['quantiles'][q2]
            
            # Linear interpolation
            weight = (q - q1) / (q2 - q1)
            return v1 + weight * (v2 - v1)
        
        def binary_search(target_coverage: float, is_upper: bool) -> float:
            left, right = 0.0, 1.0
            for _ in range(50):  # Maximum iterations for binary search
                mid = (left + right) / 2
                predicted_quantile = interpolate_quantile(mid)
                
                if is_upper:
                    current_coverage = np.mean(self.y_test.numpy() <= predicted_quantile.detach().numpy())
                else:
                    current_coverage = np.mean(self.y_test.numpy() >= predicted_quantile.detach().numpy())
                
                if abs(current_coverage - target_coverage) < 1e-3:
                    return mid
                elif current_coverage < target_coverage:
                    right = mid
                else:
                    left = mid
            
            return mid
        
        # Target coverage for each tail is (1 - coverage)/2
        tail_coverage = (1 - coverage) / 2
        lower_quantile = binary_search(tail_coverage, False)
        upper_quantile = binary_search(1 - tail_coverage, True)
        
        return lower_quantile, upper_quantile
    
    def inverse_transform(self, conformal_intervals: Dict) -> Dict:
        """
        Inverse transform the conformal prediction intervals to the original space.
        
        Args:
            conformal_intervals: Dictionary containing conformal prediction intervals
        
        Returns:
            Dictionary containing conformal prediction intervals in the original space
        """
        # Inverse transform the conformal intervals
        conformal_lower = self.scaler.inverse_transform(conformal_intervals['conformal_lower'])
        conformal_upper = self.scaler.inverse_transform(conformal_intervals['conformal_upper'])
        
        return {'conformal_lower': conformal_lower, 'conformal_upper': conformal_upper}


    def predict(self, method: str = 'absolute') -> Dict[str, np.ndarray]:
        """
        Compute prediction intervals using conformal prediction.
        
        Args:
            X: Input features
            method: Conformity score method
            
        Returns:
            Dictionary containing predicted intervals
        """

        conformal_intervals = self.predict_quantile()
        lower_q, upper_q = self._find_equivalent_quantile(conformal_intervals)
        conformal_intervals = self.inverse_transform(conformal_intervals)
        
        return {'conformal_intervals': conformal_intervals,
                'equivalent_quantiles': (lower_q, upper_q)}

def main():
    # Configurations
    CSTR_sim_config = SimulationConfig(n_simulations=100, T=101, tsim=500, noise_percentage=0.01)
    Biop_sim_config = SimulationConfig(n_simulations=10, T=20, tsim=240)
    
    
    training_config = TrainingConfig(
        batch_size=48,
        num_epochs=200,
        learning_rate=0.0031,
        time_step=10,
        horizon=5,
        weight_decay=0.01,
        factor=0.1,
        patience=58,
        delta = 0.042,
        train_test_split=0.6,
        test_val_split=0.8,
    )
    LSTM_Config = LSTMConfig(
        hidden_dim = 64,
        num_layers=8,
        dropout = 0.2,
        bidirectional=False,
        norm_type = None,
    )
    CNN_Config = CNNConfig(
        conv_channels = [16, 32],
        kernel_sizes = [5, 3],
        fc_dims = [101, 128],
        dropout = 0.1
        )
    
    TF_Config = TFConfig(
        num_layers = 2,
        hidden_dim = 64,
        d_model = 64,
        num_heads = 4,
        dim_feedforward = 128,
        dropout = 0.2,
    )
    
    MLP_Config = MLPConfig(
        hidden_dim = 64,
        num_layers = 4,
        dropout = 0.2,
        activation = 'ReLU'
    )
    
    MLR_Config = MLRConfig(
        dropout = 0.2
    )
    # Initialize components

    simulator = CSTRSimulator(CSTR_sim_config)
    # simulator = BioProcessSimulator(Biop_sim_config)
    # Get data
    simulation_results, noiseless_sim = simulator.run_multiple_simulations()
    converter = CSTRConverter()
    # converter = BioprocessConverter()
    features, targets = converter.convert(simulation_results)
    noiseless_results, _ = converter.convert(noiseless_sim)
    
    data_processor = DataProcessor(training_config, features, targets)
    # Prepare data
    (train_loader, test_loader, val_loader, X_train, X_test, X_val,  
     y_train, y_test, y_val, X, y) = data_processor.prepare_data()

    # (train_loader, test_loader, X_train, X_test, 
    #  y_train, y_test) = data_processor.prepare_data_ANNs(features, targets)
    # Initialize model (example with StandardLSTM)
    # model = CNN(
    #     config=CNN_Config,
    #     input_dim=X_train.shape[2],
    #     output_dim=y_train.shape[1],
    #     var = True
    # )
    quantiles = [0.25, 0.5, 0.75]
    
    Encoder_Config = LSTMConfig(
        hidden_dim = 64,
        num_layers=4,
        dropout = 0.2,
        bidirectional=True,
        norm_type = 'layer',
    )
    
    Decoder_Config = LSTMConfig(
        hidden_dim = 64,
        num_layers=4,
        dropout = 0.2,
        bidirectional=True,
        norm_type = 'layer',
    )
    
    # encoder = EncoderLSTM(Encoder_Config, input_dim=X_train.shape[2])
    # decoder = DecoderLSTM(Decoder_Config, hidden_dim = Encoder_Config.hidden_dim * 2 if Encoder_Config.bidirectional else Encoder_Config.hidden_dim,
    #                       output_dim=y_train.shape[2], horizon = training_config.horizon)
    
    # model = EncoderDecoder(Encoder_Config, Decoder_Config, encoder, decoder)

    encoder_config = TFConfig(
        num_layers = 4,
        hidden_dim = 64,
        d_model = 64,
        num_heads=4,
        dim_feedforward=128,
        dropout=0.2,
    )
    
    decoder_config = TFConfig(
        num_layers = 4,
        hidden_dim = 64,
        d_model = 64,
        num_heads=4,
        dim_feedforward=128,
        dropout=0.2,
    )
    
    model = EcDcTransformer(encoder_config, decoder_config, input_dim=X_train.shape[2], horizon = training_config.horizon,
                            output_dim=y_train.shape[2])
    
    # y_train of shape (time_steps, horizon, features)
    # model = LSTM(
    #     config=LSTM_Config,
    #     input_dim=X_train.shape[2],
    #     output_dim=y_train.shape[2],
    #     horizon = training_config.horizon,
    #     var = True,
    # )
    
    # print('X_train shape:', X_train.shape[2])
    # print('outptu_dim:', y_train.shape[1])
    # model = MLP(
    #     config = MLP_Config,
    #     input_dim=X_train.shape[1]*X_train.shape[2],
    #     output_dim=y_train.shape[2], 
    #     horizon = training_config.horizon,
    #     var = True
    # )
    # model = TransformerEncoder(
    #     config=TF_Config,
    #     input_dim=X_train.shape[2],
    #     output_dim=y_train.shape[1],
    #     var = True
    # )
    
    # model = MLR(
    #     config=MLR_Config,
    #     input_dim=X_train.shape[1],
    #     output_dim=y_train.shape[1],
    #     var = True
    # )

    # Train model
    criterion = nn.MSELoss()
    # criterion = QuantileLoss(quantiles)
    # criterion = EnhancedQuantileLoss(quantiles, smoothness_lambda=0.1)
    # criterion = nn.GaussianNLLLoss()
    
    trainer = ModelTrainer(model, training_config)
    model, history, avg_loss = trainer.train(train_loader, test_loader, val_loader, criterion)
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
    # scaler = data_processor.target_scaler
    
    # train_pred = data_processor.quantile_invert(train_pred, quantiles)
    # test_pred = data_processor.quantile_invert(test_pred, quantiles)
    
    with torch.no_grad():
        preds = model(X)
    # preds = data_processor.quantile_invert(pred, quantiles)
    
    mean = preds.detach().numpy()
    means = data_processor.reconstruct_sequence(mean, False)
    # var = vars.detach().numpy()
    # var = data_processor.reconstruct_sequence(var, True)
    means = data_processor.target_scaler.inverse_transform(means)
    # rescaled_pred = data_processor.rescale_predictions(mean, var)
    # means = rescaled_pred[0]
    # variances = rescaled_pred[1]
    
    # mc_predictor = MC_Prediction(model, num_samples=100)
    # train_pred, train_var = mc_predictor.predict(X_train.to(training_config.device))
    # test_pred, test_var = mc_predictor.predict(X_test.to(training_config.device))
    
    # train_pred = data_processor.target_scaler.inverse_transform(train_pred)
    # test_pred = data_processor.target_scaler.inverse_transform(test_pred)
    # train_var = data_processor.target_scaler.inverse_transform(train_var)
    # test_var = data_processor.target_scaler.inverse_transform(test_var)

    # Visualize results
    # feature_names = ['c_x', 'c_n', 'c_q']
    feature_names = ['conc', 'temp']
    action_names = ['inlet temp', 'feed conc', 'coolant temp']
    visualizer = Visualizer()
    sequence_length = training_config.time_step
    time_horizon = training_config.horizon
    # For simplicity we will plot the first simulation
    visualizer.plot_preds(means, features,
                                noiseless_results,
                                sequence_length,
                                time_horizon, feature_names,
                                num_simulations = 10,
                                train_test_split = 0.6, test_val_split = 0.8)
    # visualizer.plot_loss(history)
    # visualizer.plot_loss_loss(history)
    # print(features.shape) # 100, 60 (time steps, features)
    actions = features[:, -int(features.shape[1] - targets.shape[1]):]
    # print(actions.shape)
    # visualizer.plot_actions(actions, action_names, num_simulations = 10)
    model_class = LSTM
    trainer_class = ModelTrainer
    
    CNN_config_bounds = {
        # Training config bounds
        'batch_size': (2, 50) if isinstance(simulator, CSTRSimulator) else (2, 10),
        'num_epochs': (50, 500),
        'learning_rate': (0.0001, 0.1),
        'time_step': (2, 50) if isinstance(simulator, CSTRSimulator) else (2, 10),
        'horizon': (1, 10),
        'weight_decay': (1e-6, 0.1),
        'factor': (0.1, 0.99),
        'patience': (5, 100),
        'delta': (1e-6, 0.1),      
        
        # CNN specific bounds - much wider ranges
        'conv_channels': [(8, 128), (16, 256)],  # Much wider range for channel sizes
        'kernel_sizes': [(2, 9), (2, 7)],       # More kernel size options
        'fc_dims': [(32, 512), (64, 1024)],     # Wider range for fully connected layers
        'dropout': (0.0, 0.9)                    # Full range of dropout values
    }
    
    LSTM_ConfigBounds = {
        # Training config bounds
        'batch_size': (2, 50) if isinstance(simulator, CSTRSimulator) else (2, 10),
        'num_epochs': (50, 500),
        'learning_rate': (0.0001, 0.1),
        'time_step': (2, 50) if isinstance(simulator, CSTRSimulator) else (2, 10),
        'horizon': (1, 10),
        'weight_decay': (1e-6, 0.1),
        'factor': (0.1, 0.99),
        'patience': (5, 100),
        'delta': (1e-6, 0.1),   
        
        # LSTM specific bounds
        'hidden_dim': (32, 512),
        'num_layers': (1, 50),
        'dropout': (0.1, 0.9),
        'bidirectional': (0, 1),
        'norm_type': (0, 2),
    }
    
    MLP_config_bounds = {
        # Training config bounds
        'batch_size': (2, 50) if isinstance(simulator, CSTRSimulator) else (2, 10),
        'num_epochs': (50, 500),
        'learning_rate': (0.0001, 0.1),
        'time_step': (2, 50) if isinstance(simulator, CSTRSimulator) else (2, 10),
        'horizon': (1, 10),
        'weight_decay': (1e-6, 0.1),
        'factor': (0.1, 0.99),
        'patience': (5, 100),
        'delta': (1e-6, 0.1),   
        
        # MLP specific bounds
        'hidden_dim': (32, 512),
        'num_layers': (1, 50),
        'dropout': (0.1, 0.9),
        'activation': (0, 2),
    }
    
    # First set the bounds for LSTM Model
    TF_ConfigBounds = {
            'batch_size': (2, 50) if isinstance(simulator, CSTRSimulator) else (2, 10),
            'num_epochs': (50, 200),
            'learning_rate': (0.0001, 0.1),
            'time_step': (2, 50) if isinstance(simulator, CSTRSimulator) else (2, 10),
            'horizon': (1, 10),
            'weight_decay': (1e-6, 0.1),
            'factor': (0.1, 0.99),
            'patience': (5, 100),
            'delta': (1e-6, 0.1),   
            
            # LSTM specific bounds
            'num_layers': (1, 15),
            'hidden_dim': (32, 256),
            'd_model': (32, 256),
            'num_heads': (1, 8),
            'dim_feedforward': (32, 256),
            'dropout': (0.1, 0.9),
    }
    
    MLR_ConfigBounds = {
            'batch_size': (2, 50) if isinstance(simulator, CSTRSimulator) else (2, 10),
            'num_epochs': (50, 200),
            'learning_rate': (0.0001, 0.1),
            'time_step': (2, 50) if isinstance(simulator, CSTRSimulator) else (2, 10),
            'horizon': (1, 10),
            'weight_decay': (1e-6, 0.1),
            'factor': (0.1, 0.99),
            'patience': (5, 100),
            'delta': (1e-6, 0.1), 
            
            'dropout': (0.1, 0.9),
    }
        
    
    optimizer = ModelOptimisation(model_class, CSTR_sim_config, training_config, LSTM_Config,
                                  config_bounds=LSTM_ConfigBounds, simulator=CSTRSimulator, converter=CSTRConverter, 
                                  data_processor=DataProcessor, trainer_class=trainer_class, iters = 30, variance=True)
    path = 'best_model.pth'
    best_params, best_loss = optimizer.optimise(path)
    
    # checkpoint = torch.load('best_model.pth')
    # print(checkpoint.keys())
    # model = model_class(checkpoint['model_config'], 
    #             input_dim=X_train.shape[2],
    #             output_dim=y_train.shape[1],
    #             quantiles = quantiles)
    
    # model.load_state_dict(checkpoint['model_state_dict'])    
    # model.eval()
    # with torch.no_grad():
    #     if isinstance(criterion, nn.GaussianNLLLoss):
    #         optimised_train_pred, optimised_train_var = model(X_train.to(training_config.device))
    #         optimised_test_pred, optimised_test_var = model(X_test.to(training_config.device))
    #     else:
    #         optimised_train_pred = model(X_train.to(training_config.device)).cpu().numpy()
    #         optimised_test_pred = model(X_test.to(training_config.device)).cpu().numpy()

    # # Inverse transform predictions
    # scaler = data_processor.target_scaler
    
    # inverse_transformer = QuantileTransform(quantiles, scaler)
    # optimised_train_pred = inverse_transformer.inverse_transform(optimised_train_pred)
    # optimised_test_pred = inverse_transformer.inverse_transform(optimised_test_pred)
    
    
    # optimised_train_pred = data_processor.target_scaler.inverse_transform(optimised_train_pred)
    # optimised_test_pred = data_processor.target_scaler.inverse_transform(optimised_test_pred)
    # optimised_train_var = data_processor.target_scaler.inverse_transform(optimised_train_var)
    # optimised_test_var = data_processor.target_scaler.inverse_transform(optimised_test_var)
    
    
    # y_train_orig = data_processor.target_scaler.inverse_transform(y_train)
    # y_test_orig = data_processor.target_scaler.inverse_transform(y_test)
    
    # mc_predictor = MC_Prediction(model, training_config, num_samples=100)
    # optimised_train_pred, optimised_train_var = mc_predictor.predict(X_train.to(training_config.device))
    # optimised_test_pred, optimised_test_var = mc_predictor.predict(X_test.to(training_config.device))
    
    # optimised_train_pred = data_processor.target_scaler.inverse_transform(optimised_train_pred)
    # optimised_test_pred = data_processor.target_scaler.inverse_transform(optimised_test_pred)
    # optimised_train_var = data_processor.target_scaler.inverse_transform(optimised_train_var)
    # optimised_test_var = data_processor.target_scaler.inverse_transform(optimised_test_var)

    # Visualize results
    # visualizer = Visualizer()
    # visualizer.plot_predictions(optimised_train_pred, optimised_test_pred, y_train_orig, y_test_orig, feature_names, CSTR_sim_config.n_simulations)
    # visualizer.plot_loss(history)
    
    # print('best loss', best_loss)
    # conformal = ConformalQuantile(model, scaler, X_test, y_test, alpha=0.25)
    # results = conformal.predict()
    # Plot the training data with the uncertainty from quantiles, and then the conformal intervals on the test data
    # visualizer.plot_conformal(train_pred[0.5], test_pred[0.5], y_train_orig, y_test_orig, results, feature_names, CSTR_sim_config.n_simulations)
    
    flops = FlopCountAnalysis(model, X_train)
    print(flops)
    print(flops.total())
    print(flops.by_module())
if __name__ == "__main__":
    main()