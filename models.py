import torch
import torch.nn as nn
from base import TrainingConfig, BaseModel, LSTMConfig, CNNConfig, MLPConfig
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
import torch.nn.functional as F
from torch.autograd import Variable
import math

# LSTMs

class LSTM(BaseModel):
    """Standard LSTM implementation"""
    def __init__(self, config: LSTMConfig, input_dim: int, output_dim: int, quantiles: Optional[List[float]] = None, monte_carlo: Optional[bool] = False, var: Optional[bool] = False):
        """Initialise the LSTM model
        
        config: LSTMConfig, configuration for LSTM model
        input_dim: int, input dimension
        output_dim: int, output dimension
        quantiles: List[float], quantiles for quantile regression
        monte_carlo: bool, initialise for Monte Carlo Dropout
        var: bool, initialise for negative log likelihood loss function
        
        """
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.quantiles = quantiles
        self.monte_carlo = monte_carlo
        self.var = var
        
        # Build the LSTM output Dim
        lstm_output_dim = self.config.hidden_dim * 2 if self.config.bidirectional else self.config.hidden_dim
        # If the config specifies BatchNorm or Layer Norm, add it to the model
        self.BatchNorm = nn.BatchNorm1d(lstm_output_dim) if getattr(self.config.norm_type, 'batch', False) else None
        self.LayerNorm = nn.LayerNorm(lstm_output_dim) if getattr(self.config.norm_type, 'layer', False) else None
        
        # Normalise the first layer if BatchNorm or LayerNorm is specified
        self.input_bn = nn.BatchNorm1d(input_dim) if getattr(self.config.norm_type, 'batch', False) else None
        self.input_ln = nn.LayerNorm(input_dim) if getattr(self.config.norm_type, 'layer', False) else None
        self.input_do = nn.Dropout(p=self.config.dropout) if monte_carlo else None
        # If quantiles are specified, initialise the quantile model
        if self.quantiles is not None:
            self.output_dim = output_dim * len(quantiles)
            
        # Build the first LSTM Layer input dim -> hidden dim
        
        self.first_lstm = nn.LSTM(
            input_dim, 
            self.config.hidden_dim, 
            num_layers=1,
            batch_first=True, 
            dropout=self.config.dropout, 
            bidirectional=self.config.bidirectional
        )
        
        # Normalise the layer if this is chosen
        self.first_bn = nn.BatchNorm1d(lstm_output_dim) if getattr(self.config.norm_type, 'batch', False) else None
        self.first_ln = nn.LayerNorm(lstm_output_dim) if getattr(self.config.norm_type, 'layer', False) else None
        self.first_do = nn.Dropout(p=self.config.dropout) if monte_carlo else None
        
        # Build the other layers
        self.lstm_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for _ in range(self.config.num_layers - 1):
            lstm = nn.LSTM(
                lstm_output_dim,
                self.config.hidden_dim,
                num_layers = 1,
                batch_first=True,
                dropout=self.config.dropout,
                bidirectional=self.config.bidirectional
            )
            # self.lstm_layers.append(lstm)
            
            if monte_carlo:
                do = nn.Dropout(p=self.config.dropout)
                self.dropout_layers.append(do)
            
            if getattr(self.config.norm_type, 'batch', False):
                bn = nn.BatchNorm1d(lstm_output_dim)
                self.bn_layers.append(bn)
            else:
                self.bn_layers.append(None)
            
            if getattr(self.config.norm_type, 'layer', False):
                ln = nn.LayerNorm(lstm_output_dim)
                self.ln_layers.append(ln)
            else:
                self.ln_layers.append(None)
        
        self.fc = nn.Linear(lstm_output_dim, self.output_dim)
        
        if self.var: # If using the negative log likelihood loss function, add the log variance layer
            self.fc_logvar = nn.Linear(lstm_output_dim, self.output_dim)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_directions = 2 if self.config.bidirectional else 1
        # As LSTM is layerwise, needs to be 1 or 2 depending on bidirectional
        h0 = torch.zeros(num_directions, x.size(0), 
                        self.config.hidden_dim).to(self.config.device)
        c0 = torch.zeros(num_directions, x.size(0), 
                        self.config.hidden_dim).to(self.config.device)
        
        # Pass the input through the first normalisation if it exists
        if self.input_bn is not None:
            x = self.input_bn(x)
        if self.input_ln is not None:
            x = self.input_ln(x)
        if self.input_do is not None:
            x = self.input_do(x)
                    
        # Pass through the first layer
        lstm_out, _ = self.first_lstm(x, (h0, c0))


        # Normalise or dropout the output of first layer if it is specified
        if self.first_bn is not None:
            lstm_out = self.first_bn(lstm_out)
        if self.first_ln is not None:
            lstm_out = self.first_ln(lstm_out)
        if self.first_do is not None:
            lstm_out = self.first_do(lstm_out)
            
        # Apply the other layers
        for lstm, bn_layer, ln_layer, do_layer in zip(self.lstm_layers, self.bn_layers, self.ln_layers, self.dropout_layers):
            h0 = torch.zeros(self.config.num_layers-1, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
            c0 = torch.zeros(self.config.num_layers-1, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
            
            lstm_out, _ = lstm(lstm_out, (h0, c0))
            # Apply the batch,layer norm and dropout if specified
            if bn_layer is not None:
                lstm_out = bn_layer(lstm_out)
            if ln_layer is not None:
                lstm_out = ln_layer(lstm_out)
            if self.monte_carlo:
                lstm_out = do_layer(lstm_out)
        
        # For the basic LSTM just return the final prediction
        x = self.fc(lstm_out[:, -1, :])
        
        if self.monte_carlo: # Dropout from the output
        # Pass the output 
            x = nn.Dropout(p=self.config.dropout)(lstm_out[:, -1, :])
            return self.fc(x)
            
        
        if self.var:
            # Pass the log variance through the exponential function to get the variance
            # from the fully connected layer
            var = torch.exp(self.fc_logvar(lstm_out[:, -1, :]))
            x = self.fc(lstm_out[:, -1, :])
            return x, var
        
        if self.quantiles is not None:
            # Output the quantiles
            return x.view(-1, self.output_dim // len(self.quantiles), len(self.quantiles))
        
        return x

class StandardLSTM(BaseModel):
    """Standard LSTM implementation"""
    def __init__(self, config: LSTMConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        
        lstm_output_dim = self.config.hidden_dim * 2 if self.config.bidirectional else self.config.hidden_dim
        self.lstm = nn.LSTM(
            input_dim, 
            self.config.hidden_dim, 
            self.config.num_layers,  # Make sure this line matches exactly
            batch_first=True, 
            dropout=self.config.dropout, 
            bidirectional=self.config.bidirectional
        )
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.bidirectional:
            h0 = torch.zeros(self.config.num_layers * 2, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
            c0 = torch.zeros(self.config.num_layers * 2, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
        else:
            h0 = torch.zeros(self.config.num_layers, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
            c0 = torch.zeros(self.config.num_layers, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
            
        lstm_out, _ = self.lstm(x, (h0, c0))
        return self.fc(lstm_out[:, -1, :])

class RSQuantileLSTM(BaseModel):
    """LSTM with quantile regression, residual connections, and batch normalization"""
    def __init__(self, config: LSTMConfig, input_dim: int, output_dim: int, quantiles: List[float]):
        super().__init__(config)
        self.quantiles = quantiles
        self.output_dim = output_dim * len(quantiles)
        
        # Calculate output dimension for each LSTM layer
        self.lstm_output_dim = self.config.hidden_dim * 2 if self.config.bidirectional else self.config.hidden_dim
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_dim) if getattr(config, 'use_batch_norm', False) else None
        
        # First LSTM layer (input dim → hidden dim)
        self.first_lstm = nn.LSTM(
            input_dim,
            self.config.hidden_dim,
            1,
            batch_first=True,
            bidirectional=self.config.bidirectional
        )
        
        # Optional BatchNorm after first layer
        self.first_bn = nn.BatchNorm1d(self.lstm_output_dim) if getattr(config, 'use_batch_norm', False) else None
        
        # Projection layer for residual connection if input_dim != lstm_output_dim
        self.residual_projection = None
        if input_dim != self.lstm_output_dim:
            self.residual_projection = nn.Linear(input_dim, self.lstm_output_dim)
        
        # Additional LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for _ in range(config.num_layers - 1):
            lstm = nn.LSTM(
                self.lstm_output_dim,
                self.config.hidden_dim,
                1,
                batch_first=True,
                bidirectional=self.config.bidirectional
            )
            self.lstm_layers.append(lstm)
            
            if getattr(config, 'use_batch_norm', False):
                bn = nn.BatchNorm1d(self.lstm_output_dim)
                self.bn_layers.append(bn)
            else:
                self.bn_layers.append(None)
        
        # Output layer
        self.fc = nn.Linear(self.lstm_output_dim, self.output_dim)
        
    def _apply_lstm_with_residual(self, x, lstm, bn_layer, h0, c0, residual=None):
        """Helper function to apply LSTM with residual connection and batch norm"""
        lstm_out, (hn, cn) = lstm(x, (h0, c0))
        
        # Apply batch normalization if enabled
        if bn_layer is not None:
            # Reshape for batch norm
            lstm_out_reshape = lstm_out.reshape(-1, lstm_out.size(-1))
            lstm_out_norm = bn_layer(lstm_out_reshape)
            lstm_out = lstm_out_norm.reshape(lstm_out.size())
        
        # Add residual connection if provided
        if residual is not None:
            if self.residual_projection is not None and hasattr(self, 'first_lstm') and lstm is self.first_lstm:
                residual = self.residual_projection(residual)
            lstm_out = lstm_out + residual
        
        return lstm_out, (hn, cn)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Apply input batch norm if enabled
        if self.input_bn is not None:
            x_reshaped = x.reshape(-1, x.size(-1))
            x = self.input_bn(x_reshaped)
            x = x.reshape(batch_size, seq_len, -1)
        
        # Initial hidden states
        h_size = 2 if self.config.bidirectional else 1
        h0 = torch.zeros(h_size, batch_size, self.config.hidden_dim).to(self.config.device)
        c0 = torch.zeros(h_size, batch_size, self.config.hidden_dim).to(self.config.device)
        
        # First LSTM layer with optional residual
        residual = x  # Save input for residual connection
        x, (h0, c0) = self._apply_lstm_with_residual(x, self.first_lstm, self.first_bn, h0, c0, residual)
        
        # Apply dropout after first layer
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        
        # Additional LSTM layers
        for lstm, bn_layer in zip(self.lstm_layers, self.bn_layers):
            residual = x  # Save previous output for residual connection
            h0 = torch.zeros(h_size, batch_size, self.config.hidden_dim).to(self.config.device)
            c0 = torch.zeros(h_size, batch_size, self.config.hidden_dim).to(self.config.device)
            
            x, (h0, c0) = self._apply_lstm_with_residual(x, lstm, bn_layer, h0, c0, residual)
            
            # Apply dropout between layers
            x = F.dropout(x, p=self.config.dropout, training=self.training)
        
        # Use only the last output for prediction
        x = x[:, -1, :]
        predictions = self.fc(x)
        
        return predictions.view(-1, self.output_dim // len(self.quantiles), len(self.quantiles))

class QuantileLSTM(BaseModel):
    """LSTM with quantile regression capabilities"""
    def __init__(self, config: LSTMConfig, input_dim: int, output_dim: int, quantiles: List[float]):
        super().__init__(config)
        self.quantiles = quantiles
        self.output_dim = output_dim * len(quantiles)
        
        lstm_output_dim = self.config.hidden_dim * 2 if self.config.bidirectional else self.config.hidden_dim

        self.lstm = nn.LSTM(
            input_dim, self.config.hidden_dim, self.config.num_layers,
            batch_first=True, dropout=self.config.dropout, bidirectional=self.config.bidirectional
        )
        self.fc = nn.Linear(lstm_output_dim, self.output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.bidirectional:
            h0 = torch.zeros(self.config.num_layers * 2, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
            c0 = torch.zeros(self.config.num_layers * 2, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
        else:
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
    def __init__(self, config: LSTMConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        lstm_output_dim = self.config.hidden_dim * 2 if self.config.bidirectional else self.config.hidden_dim
        self.lstm = nn.LSTM(
            input_dim, self.config.hidden_dim, self.config.num_layers,
            batch_first=True, dropout=self.config.dropout, bidirectional=self.config.bidirectional
        )
        self.fc_mean = nn.Linear(lstm_output_dim, output_dim) # Fully connected layer for mean prediction
        self.fc_logvar = nn.Linear(lstm_output_dim, output_dim) # Fully connected layer for log variance prediction
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.bidirectional:
            h0 = torch.zeros(self.config.num_layers * 2, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
            c0 = torch.zeros(self.config.num_layers * 2, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
        else:
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
    def __init__(self, config: LSTMConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        lstm_output_dim = self.config.hidden_dim * 2 if self.config.bidirectional else self.config.hidden_dim
        self.lstm = nn.LSTM(
            self.input_dim, self.config.hidden_dim, self.config.num_layers,
            batch_first=True, dropout=self.config.dropout, bidirectional=self.config.bidirectional
        )
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.bidirectional:
            h0 = torch.zeros(self.config.num_layers * 2, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
            c0 = torch.zeros(self.config.num_layers * 2, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
        else:
            h0 = torch.zeros(self.config.num_layers, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
            c0 = torch.zeros(self.config.num_layers, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        dropout_out = self.dropout(lstm_out[:, -1, :])
        return self.fc(dropout_out)

# https://colab.research.google.com/github/PawaritL/BayesianLSTM/blob/master/Energy_Consumption_Predictions_with_Bayesian_LSTMs_in_PyTorch.ipynb#scrollTo=OgWyOffPbO0b

class BayesianLSTM(nn.Module):

    def __init__(self, n_features, output_length, batch_size):

        super(BayesianLSTM, self).__init__()

        self.batch_size = batch_size # user-defined

        self.hidden_size_1 = 128 # number of encoder cells (from paper)
        self.hidden_size_2 = 32 # number of decoder cells (from paper)
        self.stacked_layers = 2 # number of (stacked) LSTM layers for each stage
        self.dropout_probability = 0.5 # arbitrary value (the paper suggests that performance is generally stable across all ranges)

        self.lstm1 = nn.LSTM(n_features, 
                             self.hidden_size_1, 
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.stacked_layers,
                             batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size_2, output_length)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        output = output[:, -1, :] # take the last decoder cell's outputs
        y_pred = self.fc(output)
        return y_pred
        
    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        return hidden_state, cell_state
    
    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        return hidden_state, cell_state
    
    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)

    def predict(self, X):
        return self(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()

# Convolutional Neural Networks (CNNs)

class CNN(BaseModel):
    """Unified CNN implementation supporting multiple uncertainty estimation methods"""
    def __init__(self, config: CNNConfig, input_dim: int, output_dim: int, 
                 quantiles: Optional[List[float]] = None, 
                 monte_carlo: Optional[bool] = False, 
                 var: Optional[bool] = False):
        """Initialize CNN model with various uncertainty estimation capabilities
        
        Args:
            config: CNNConfig object containing model parameters
            input_dim: Input dimension
            output_dim: Output dimension
            quantiles: List of quantiles for quantile regression
            monte_carlo: Whether to use Monte Carlo dropout
            var: Whether to estimate variance (for NLL loss)
        """
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.quantiles = quantiles
        self.monte_carlo = monte_carlo
        self.var = var
        
        # Adjust output dimension for quantile regression
        if self.quantiles is not None:
            self.output_dim = output_dim * len(quantiles)
        
        # Build CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_dim
        
        for out_channels, kernel_size in zip(self.config.conv_channels, self.config.kernel_sizes):
            conv_block = [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU()
            ]
            
            # Add normalization if specified
            if getattr(self.config, 'norm_type', None):
                if getattr(self.config.norm_type, 'batch', False):
                    conv_block.append(nn.BatchNorm1d(out_channels))
                if getattr(self.config.norm_type, 'layer', False):
                    conv_block.append(nn.LayerNorm([out_channels, -1]))
            
            # Add dropout for Monte Carlo
            if monte_carlo:
                conv_block.append(nn.Dropout(self.config.dropout))
                
            self.conv_layers.append(nn.Sequential(*conv_block))
            in_channels = out_channels
            
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Build FC layers
        self.fc_layers = nn.ModuleList()
        fc_input_dim = self.config.conv_channels[-1]
        
        for fc_dim in self.config.fc_dims:
            fc_block = [
                nn.Linear(fc_input_dim, fc_dim),
                nn.ReLU()
            ]
            
            if monte_carlo:
                fc_block.append(nn.Dropout(self.config.dropout))
                
            self.fc_layers.append(nn.Sequential(*fc_block))
            fc_input_dim = fc_dim
        
        # Output layers
        self.fc = nn.Linear(fc_input_dim, self.output_dim)
        if self.var:
            self.fc_logvar = nn.Linear(fc_input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input for CNN: [batch_size, features, time_step]
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # Apply FC layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        if self.var:
            # Return mean and variance for NLL loss
            var = torch.exp(self.fc_logvar(x))
            return self.fc(x), var
            
        if self.quantiles is not None:
            # Return reshaped output for quantile regression
            return self.fc(x).view(-1, self.output_dim // len(self.quantiles), len(self.quantiles))
            
        # Standard or Monte Carlo dropout output
        return self.fc(x)

class StandardCNN(BaseModel):
    """ Standard CNN Implementation """
    def __init__(self, config: CNNConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        
        # Build CNN layers dynamically
        self.conv_layers = nn.ModuleList()
        self.output_dim = output_dim
        in_channels = input_dim
        
        for out_channels, kernel_size in zip(self.config.conv_channels, self.config.kernel_sizes):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels)
            ))
            in_channels = out_channels
            
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Build FC layers dynamically
        self.fc_layers = nn.ModuleList()
        fc_input_dim = self.config.conv_channels[-1]
        
        for i, fc_dim in enumerate(self.config.fc_dims):
            self.fc_layers.append(nn.Linear(fc_input_dim, fc_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(self.config.dropout))
            fc_input_dim = fc_dim
            
        self.fc_layers.append(nn.Linear(fc_input_dim, output_dim))
        

    def forward(self, x):
        # Input shape: [batch_size, time_step, features]
        # CNN expects: [batch_size, features, time_step]
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        return x
    
class NLL_CNN(BaseModel):
    """ CNN with NLL Likelihood Loss Function, outputs mean and log variance predictions """
    def __init__(self, config: CNNConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        
        # Build CNN layers dynamically
        self.conv_layers = nn.ModuleList()
        self.output_dim = output_dim
        in_channels = input_dim
        
        for out_channels, kernel_size in zip(self.config.conv_channels, self.config.kernel_sizes):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels)
            ))
            in_channels = out_channels
            
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Build shared FC layers dynamically
        self.shared_fc_layers = nn.ModuleList()
        fc_input_dim = self.config.conv_channels[-1]
        
        for i, fc_dim in enumerate(self.config.fc_dims):
            self.shared_fc_layers.append(nn.Linear(fc_input_dim, fc_dim))
            self.shared_fc_layers.append(nn.ReLU())
            self.shared_fc_layers.append(nn.Dropout(self.config.dropout))
            fc_input_dim = fc_dim
            
        # Separate output layers for mean and log variance
        self.fc_mean = nn.Linear(fc_input_dim, output_dim)
        self.fc_logvar = nn.Linear(fc_input_dim, output_dim)

    def forward(self, x):
        # Input shape: [batch_size, time_step, features]
        # CNN expects: [batch_size, features, time_step]
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # Apply shared FC layers
        for fc_layer in self.shared_fc_layers:
            x = fc_layer(x)
        
        # Generate mean and log variance predictions
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        var = torch.exp(logvar)
        
        return mean, var

class MC_CNN(BaseModel):
    """ Standard CNN Implementation with Monte Carlo Dropout """
    def __init__(self, config: CNNConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        
        # Build CNN layers dynamically
        self.conv_layers = nn.ModuleList()
        self.output_dim = output_dim
        in_channels = input_dim
        
        for out_channels, kernel_size in zip(self.config.conv_channels, self.config.kernel_sizes):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Dropout(self.config.dropout)  # Add dropout layer
            ))
            in_channels = out_channels
            
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Build FC layers dynamically
        self.fc_layers = nn.ModuleList()
        fc_input_dim = self.config.conv_channels[-1]
        
        for i, fc_dim in enumerate(self.config.fc_dims):
            self.fc_layers.append(nn.Linear(fc_input_dim, fc_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(self.config.dropout))  # Add dropout layer
            fc_input_dim = fc_dim
            
        self.fc_layers.append(nn.Linear(fc_input_dim, output_dim))

    def forward(self, x):
        # Input shape: [batch_size, time_step, features]
        # CNN expects: [batch_size, features, time_step]
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        return x

class QuantileCNN(BaseModel):
    """CNN with quantile regression capabilities"""
    def __init__(self, config: CNNConfig, input_dim: int, output_dim: int, quantiles: List[float]):
        super().__init__(config)
        self.quantiles = quantiles
        self.conv_layers = nn.ModuleList()
        self.output_dim = output_dim * len(quantiles)
        in_channels = input_dim
        
        for out_channels, kernel_size in zip(self.config.conv_channels, self.config.kernel_sizes):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels)
            ))
            in_channels = out_channels
            
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Build FC layers dynamically
        self.fc_layers = nn.ModuleList()
        fc_input_dim = self.config.conv_channels[-1]
        
        for i, fc_dim in enumerate(self.config.fc_dims):
            self.fc_layers.append(nn.Linear(fc_input_dim, fc_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(self.config.dropout))
            fc_input_dim = fc_dim
            
        self.fc_layers.append(nn.Linear(fc_input_dim, self.output_dim))

    def forward(self, x):
        # Input shape: [batch_size, time_step, features]
        # CNN expects: [batch_size, features, time_step]
        x = x.transpose(1, 2)
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        return x.view(-1, self.output_dim // len(self.quantiles), len(self.quantiles))

# Multi-Layer Perceptrons (ANNs)

activations = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid", "Softmax", "LogSoftmax"]

class MLP(BaseModel):
    """Unified MLP implementation supporting multiple uncertainty estimation methods"""
    def __init__(self, config: MLPConfig, input_dim: int, output_dim: int,
                 quantiles: Optional[List[float]] = None,
                 monte_carlo: Optional[bool] = False,
                 var: Optional[bool] = False):
        """Initialize MLP model with various uncertainty estimation capabilities
        
        Args:
            config: MLPConfig object containing model parameters
            input_dim: Input dimension
            output_dim: Output dimension
            quantiles: List of quantiles for quantile regression
            monte_carlo: Whether to use Monte Carlo dropout
            var: Whether to estimate variance (for NLL loss)
        """
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.quantiles = quantiles
        self.monte_carlo = monte_carlo
        self.var = var
        
        # Adjust output dimension for quantile regression
        if self.quantiles is not None:
            self.output_dim = self.output_dim * len(quantiles)
            
        # Validate activation function
        assert config.activation in activations, "Activation function not supported"
        self.activation = getattr(nn, config.activation)()
        
        # Build layers
        layers = []
        current_dim = self.input_dim
        
        # Input layer
        layers.extend([
            nn.Linear(current_dim, config.hidden_dim),
            self.activation
        ])
        
        if monte_carlo:
            layers.append(nn.Dropout(config.dropout))
            
        current_dim = config.hidden_dim
        
        # Hidden layers
        for _ in range(config.num_layers - 2):
            layers.extend([
                nn.Linear(current_dim, config.hidden_dim),
                self.activation
            ])
            
            if monte_carlo:
                layers.append(nn.Dropout(config.dropout))
                
            current_dim = config.hidden_dim
            
        self.layers = nn.Sequential(*layers)
        
        # Output layers
        self.fc = nn.Linear(current_dim, self.output_dim)
        if self.var:
            self.fc_logvar = nn.Linear(current_dim, self.output_dim)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        
        if self.var:
            # Return mean and variance for NLL loss
            var = torch.exp(self.fc_logvar(x))
            return self.fc(x), var
            
        if self.quantiles is not None:
            gramah = self.fc(x)
            # Return reshaped output for quantile regression
            return self.fc(x).view(-1, self.output_dim // len(self.quantiles), len(self.quantiles))
            
        # Standard or Monte Carlo dropout output
        return self.fc(x)

class QuantileMLP(BaseModel):
    """Quantile Multi-Layer Perceptron
        
        config: MLPConfig, configuration for MLP model
        input_dim: int, input dimension
        output_dim: int, output dimension
           
    """
    def __init__(
        self, config: MLPConfig, input_dim: int, output_dim: int, quantiles: List[float]):
        super().__init__(config)
        self.config = config
        self.quantiles = quantiles
        self.input_dim = input_dim
        self.hidden_dim = self.config.hidden_dim
        self.output_dim = self.output_dim * len(self.quantiles)
        assert self.config.activation in activations, "Activation function not supported"
        self.config.activation = getattr(nn, self.config.activation)()
        # Input Layers 
        layers = [
            nn.Linear(input_dim, self.config.hidden_dim),
            self.activation,
            nn.Dropout(self.config.dropout)
        ]
        # Hidden Layers
        for _ in range(self.config.num_layers-2):
            layers += [
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                self.activation,
                nn.Dropout(self.config.dropout)
            ]
        # Output Layer
        layers += [
            nn.Linear(self.config.hidden_dim, self.output_dim)
        ]
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.layers(x)
        return out.view(-1, self.output_dim // len(self.quantiles), len(self.quantiles))

# Linear Regression

class MLR(BaseModel):
    """Multi Linear Regression Model"""
    def __init__(self, config: TrainingConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
    
# Transformers
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

# Encoder Only Model
class TransformerEncoder(BaseModel):
    def __init__(self, config: TrainingConfig, input_dim: int, output_dim: int,
                 quantiles: Optional[List[float]] = None, 
                 monte_carlo: Optional[bool] = False, var: Optional[bool] = False):
        super().__init__(config)
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.quantiles = quantiles
        self.monte_carlo = monte_carlo
        self.var = var
        
        if self.quantiles is not None:
            self.output_dim = output_dim * len(quantiles)
        

        # Input projection layer
        self.input_projection = nn.Linear(self.input_dim, self.config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.config.d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.config.num_layers)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dim_feedforward, self.output_dim)
        )
        
        if self.var:            
            # self.fc_logvar = nn.Linear(self.config.d_model, self.output_dim)
            self.fc_logvar = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.dim_feedforward),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.dim_feedforward, self.output_dim)
            )

    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Create mask for transformer (optional, useful for variable length sequences)
        mask = None
        
        # Apply transformer encoder
        output = self.transformer_encoder(x, mask)
        
        # Take the last sequence element for prediction
        output_seq = output[:, -1, :]
        
        # Project to output dimension
        output = self.output_layer(output_seq)
        
        
        if self.var:
            var = torch.exp(self.fc_logvar(output_seq))
            return output, var
        
        if self.monte_carlo:
            output = nn.Dropout(p=self.config.dropout)(output_seq)
            return self.output_layer(output)
        
        if self.quantiles is not None:
            return output.view(-1, self.output_dim // len(self.quantiles), len(self.quantiles))
        
        return output
        
# Graph Neural Networks (GNNs)

class ST_GCN(BaseModel):
    """Spatial-Temporal Graph Convolutional Network"""
    
    