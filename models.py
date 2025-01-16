import torch
import torch.nn as nn
from base import TrainingConfig, BaseModel, LSTMConfig, CNNConfig, MLPConfig
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
import torch.nn.functional as F
from torch.autograd import Variable


class StandardLSTM(BaseModel):
    """Standard LSTM implementation"""
    def __init__(self, config: LSTMConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        lstm_output_dim = self.config.hidden_dim * 2 if self.config.bidirectional else self.config.hidden_dim
        self.lstm = nn.LSTM(
            input_dim, self.config.hidden_dim, self.config.self.config.num_layers,
            batch_first=True, dropout=self.config.dropout, bidirectional=self.config.bidirectional
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

activations = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid", "Softmax", "LogSoftmax"]

class MLP(BaseModel):
    
    """Multi-Layer Perceptron
        
        config: MLPConfig, configuration for MLP model
        input_dim: int, input dimension
        output_dim: int, output dimension
           
    """
    def __init__(
        self, config: MLPConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = self.config.hidden_dim
        self.output_dim = self.output_dim
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
        return self.layers(x)

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


class MLR(BaseModel):
    """Multi Linear Regression Model"""
    def __init__(self, config: TrainingConfig, input_dim: int, output_dim: int):
        super().__init__(config)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)