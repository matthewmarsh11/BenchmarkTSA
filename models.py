import torch
import torch.nn as nn
from base import TrainingConfig, BaseModel, BaseEncoderDecoder, LSTMConfig, CNNConfig, MLPConfig, TFConfig
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass
import torch.nn.functional as F
from torch.autograd import Variable
import math

# LSTMs

class LSTM(BaseModel):
    """Standard LSTM implementation"""
    def __init__(self, config: LSTMConfig, input_dim: int, output_dim: int, horizon: int, quantiles: Optional[List[float]] = None, monte_carlo: Optional[bool] = False, var: Optional[bool] = False):
        """Initialise the LSTM model
        
        config: LSTMConfig, configuration for LSTM model
        input_dim: int, input dimension
        output_dim: int, output dimension
        horizon: int, number of time steps to predict
        quantiles: List[float], quantiles for quantile regression
        monte_carlo: bool, initialise for Monte Carlo Dropout
        var: bool, initialise for negative log likelihood loss function
        
        """
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        
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
        
        self.first_bn = nn.BatchNorm1d(lstm_output_dim) if getattr(self.config.norm_type, 'batch', False) else None
        self.first_ln = nn.LayerNorm(lstm_output_dim) if getattr(self.config.norm_type, 'layer', False) else None
        self.first_do = nn.Dropout(p=self.config.dropout) if monte_carlo else None
        
        # Additional LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                lstm_output_dim,
                self.config.hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0,  # Dropout is handled separately
                bidirectional=self.config.bidirectional
            ) for _ in range(self.config.num_layers - 1)
        ])
        
        # Normalization and dropout for additional layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(lstm_output_dim) if getattr(self.config.norm_type, 'batch', False) else None
            for _ in range(self.config.num_layers - 1)
        ])
        
        self.ln_layers = nn.ModuleList([
            nn.LayerNorm(lstm_output_dim) if getattr(self.config.norm_type, 'layer', False) else None
            for _ in range(self.config.num_layers - 1)
        ])
        
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(p=self.config.dropout) if monte_carlo else None
            for _ in range(self.config.num_layers - 1)
        ])
        
        self.fc = nn.Linear(lstm_output_dim, self.output_dim * self.horizon)
        
        if self.var: # If using the negative log likelihood loss function, add the log variance layer
            self.fc_logvar = nn.Linear(lstm_output_dim, self.output_dim * self.horizon)
            
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
            h0 = torch.zeros(num_directions, x.size(0), 
                            self.config.hidden_dim).to(self.config.device)
            c0 = torch.zeros(num_directions, x.size(0), 
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
            out = self.fc(x)
            # shape of out is [batch_size, output_dim], reshape to [batch, horizon, features]
            return out.view(-1, self.horizon, self.output_dim)
            
        
        if self.var:
            # Pass the log variance through the exponential function to get the variance
            # from the fully connected layer
            var = torch.exp(self.fc_logvar(lstm_out[:, -1, :]))
            x = self.fc(lstm_out[:, -1, :])
            return x.view(-1, self.horizon, self.output_dim), var.view(-1, self.horizon, self.output_dim)
        
        if self.quantiles is not None:
            # Output the quantiles
            return x.view(-1, self.horizon, self.output_dim // len(self.quantiles), len(self.quantiles))
        
        return x.view(-1, self.horizon, self.output_dim)

# Encoder and Decoder LSTMs

class EncoderLSTM(BaseModel):
    
    def __init__(self, config: LSTMConfig, input_dim: int,
                 monte_carlo: Optional[bool] = False):
        super().__init__(config)
        
        self.input_dim = input_dim
        self.monte_carlo = monte_carlo
        
        lstm_output_dim = self.config.hidden_dim * 2 if self.config.bidirectional else self.config.hidden_dim

        self.BatchNorm = nn.BatchNorm1d(lstm_output_dim) if getattr(self.config.norm_type, 'batch', False) else None
        self.LayerNorm = nn.LayerNorm(lstm_output_dim) if getattr(self.config.norm_type, 'layer', False) else None
        
        # Normalise the first layer if BatchNorm or LayerNorm is specified
        self.input_bn = nn.BatchNorm1d(input_dim) if getattr(self.config.norm_type, 'batch', False) else None
        self.input_ln = nn.LayerNorm(input_dim) if getattr(self.config.norm_type, 'layer', False) else None
        self.input_do = nn.Dropout(p=self.config.dropout) if monte_carlo else None

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
        
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                lstm_output_dim,
                self.config.hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0,  # Dropout is handled separately
                bidirectional=self.config.bidirectional
            ) for _ in range(self.config.num_layers - 1)
        ])
        
        # Normalization and dropout for additional layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(lstm_output_dim) if getattr(self.config.norm_type, 'batch', False) else None
            for _ in range(self.config.num_layers - 1)
        ])
        
        self.ln_layers = nn.ModuleList([
            nn.LayerNorm(lstm_output_dim) if getattr(self.config.norm_type, 'layer', False) else None
            for _ in range(self.config.num_layers - 1)
        ])
        
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(p=self.config.dropout) if monte_carlo else None
            for _ in range(self.config.num_layers - 1)
        ])
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        
        num_directions = 2 if self.config.bidirectional else 1
        if hidden is None:
            h0 = torch.zeros(num_directions, x.size(0), self.config.hidden_dim).to(self.config.device)
            c0 = torch.zeros(num_directions, x.size(0), self.config.hidden_dim).to(self.config.device)
            hidden = (h0, c0)
        
        if self.input_bn is not None:
            x = x.transpose(1,2)
            x = self.input_bn(x)
            x = x.transpose(1,2)
        if self.input_ln is not None:
            x = self.input_ln(x)
        if self.input_do is not None:
            x = self.input_do(x)
            
        out, hidden = self.first_lstm(x, (h0, c0))
        
        if self.first_bn is not None:
            out = out.transpose(1,2)
            out = self.first_bn(out)
            out = out.transpose(1,2)
        if self.first_ln is not None:
            out = self.first_ln(out)
        if self.first_do is not None:
            out = self.first_do(out)
            
        for lstm, bn_layer, ln_layer, do_layer in zip(self.lstm_layers, self.bn_layers, self.ln_layers, self.dropout_layers):
            h0 = torch.zeros(num_directions, x.size(0), self.config.hidden_dim).to(self.config.device)
            c0 = torch.zeros(num_directions, x.size(0), self.config.hidden_dim).to(self.config.device)
            
            out, hidden = lstm(out, (h0, c0))
            
            if bn_layer is not None:
                out = out.transpose(1,2)
                out = bn_layer(out)
                out = out.transpose(1,2)
            if ln_layer is not None:
                out = ln_layer(out)
            if self.monte_carlo:
                out = do_layer(out)
                
        if self.monte_carlo:
            out = nn.Dropout(p=self.config.dropout)(out)
            
        return out, hidden
        

class DecoderLSTM(BaseModel):
    
    def __init__(self, config: LSTMConfig, hidden_dim: int, output_dim: int, horizon: int, quantiles: Optional[List[float]] = None, monte_carlo: Optional[bool] = False, var: Optional[bool] = False):
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        
        
        self.quantiles = quantiles
        self.monte_carlo = monte_carlo
        self.var = var

        lstm_output_dim = self.config.hidden_dim * 2 if self.config.bidirectional else self.config.hidden_dim

        self.BatchNorm = nn.BatchNorm1d(lstm_output_dim) if getattr(self.config.norm_type, 'batch', False) else None
        self.LayerNorm = nn.LayerNorm(lstm_output_dim) if getattr(self.config.norm_type, 'layer', False) else None
        
        # Normalise the first layer if BatchNorm or LayerNorm is specified
        self.input_bn = nn.BatchNorm1d(self.hidden_dim) if getattr(self.config.norm_type, 'batch', False) else None
        self.input_ln = nn.LayerNorm(self.hidden_dim) if getattr(self.config.norm_type, 'layer', False) else None
        self.input_do = nn.Dropout(p=self.config.dropout) if monte_carlo else None

        # If quantiles are specified, initialise the quantile model
        if self.quantiles is not None:
            self.output_dim = output_dim * len(quantiles)

        # Build the first LSTM Layer input dim -> hidden dim
        
        self.first_lstm = nn.LSTM(
            self.hidden_dim, 
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
        
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                lstm_output_dim,
                self.config.hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0,  # Dropout is handled separately
                bidirectional=self.config.bidirectional
            ) for _ in range(self.config.num_layers - 1)
        ])
        
        # Normalization and dropout for additional layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(lstm_output_dim) if getattr(self.config.norm_type, 'batch', False) else None
            for _ in range(self.config.num_layers - 1)
        ])
        
        self.ln_layers = nn.ModuleList([
            nn.LayerNorm(lstm_output_dim) if getattr(self.config.norm_type, 'layer', False) else None
            for _ in range(self.config.num_layers - 1)
        ])
        
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(p=self.config.dropout) if monte_carlo else None
            for _ in range(self.config.num_layers - 1)
        ])
        
        self.fc = nn.Linear(lstm_output_dim, self.output_dim * self.horizon)
        
        if self.var: # If using the negative log likelihood loss function, add the log variance layer
            self.fc_logvar = nn.Linear(lstm_output_dim, self.output_dim * self.horizon)

    def forward(self,
                decoder_input,
                encoder_hidden):

        decoder_hidden = encoder_hidden
        x = decoder_input        

        if self.input_bn is not None:
            x = x.transpose(1,2)
            x = self.input_bn(x)
            x = x.transpose(1,2)
        if self.input_ln is not None:
            x = self.input_ln(x)
        if self.input_do is not None:
            x = self.input_do(x)
        
        # Pass through the first layer
        lstm_out, decoder_hidden = self.first_lstm(x, decoder_hidden)

        # Normalise or dropout the output of first layer if it is specified
        if self.first_bn is not None:
            lstm_out = lstm_out.transpose(1,2)
            lstm_out = self.first_bn(lstm_out)
            lstm_out = lstm_out.transpose(1,2)
        if self.first_ln is not None:
            lstm_out = self.first_ln(lstm_out)
        if self.first_do is not None:
            lstm_out = self.first_do(lstm_out)
            
        # Apply the other layers
        for lstm, bn_layer, ln_layer, do_layer in zip(self.lstm_layers, self.bn_layers, self.ln_layers, self.dropout_layers):
            
            lstm_out, decoder_hidden = lstm(lstm_out, decoder_hidden)
            # Apply the batch,layer norm and dropout if specified
            if bn_layer is not None:
                lstm_out = lstm_out.transpose(1,2)
                lstm_out = bn_layer(lstm_out)
                lstm_out = lstm_out.transpose(1,2)
            if ln_layer is not None:
                lstm_out = ln_layer(lstm_out)
            if self.monte_carlo:
                lstm_out = do_layer(lstm_out)
            

        
        # For the basic LSTM just return the final prediction
        x = self.fc(lstm_out[:, -1, :])
        
        if self.monte_carlo: # Dropout from the output
        # Pass the output 
            x = nn.Dropout(p=self.config.dropout)(lstm_out[:, -1, :])
            out = self.fc(x)
            # shape of out is [batch_size, output_dim], reshape to [batch, horizon, features]
            return out.view(-1, self.horizon, self.output_dim)
            
        
        if self.var:
            # Pass the log variance through the exponential function to get the variance
            # from the fully connected layer
            var = torch.exp(self.fc_logvar(lstm_out[:, -1, :]))
            x = self.fc(lstm_out[:, -1, :])
            return x.view(-1, self.horizon, self.output_dim), var.view(-1, self.horizon, self.output_dim)
        
        if self.quantiles is not None:
            # Output the quantiles
            return x.view(-1, self.horizon, self.output_dim // len(self.quantiles), len(self.quantiles))
        
        return x.view(-1, self.horizon, self.output_dim)
        

# Combined Encoder-Decoder Model (Can be used in combination with other models!)

class EncoderDecoder(BaseEncoderDecoder):
    
    def __init__(self, encoder_config: LSTMConfig, decoder_config: LSTMConfig, encoder_model: BaseModel, decoder_model: BaseModel):
        super().__init__(encoder_config, decoder_config)
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Encoder forward pass
        encoder_output, encoder_hidden = self.encoder_model(x)
        
        # Decoder forward pass using encoder's hidden state
        decoder_output = self.decoder_model(encoder_output, encoder_hidden)
        
        return decoder_output
        
# Convolutional Neural Networks (CNNs)

class CNN(BaseModel):
    """Unified CNN implementation supporting multiple uncertainty estimation methods"""
    def __init__(self, config: CNNConfig, input_dim: int, output_dim: int, 
                 horizon: int, quantiles: Optional[List[float]] = None, 
                 monte_carlo: Optional[bool] = False, 
                 var: Optional[bool] = False):
        """Initialize CNN model with various uncertainty estimation capabilities
        
        Args:
            config: CNNConfig object containing model parameters
            input_dim: Input dimension
            output_dim: Output dimension
            horizon: Number of time steps to predict
            quantiles: List of quantiles for quantile regression
            monte_carlo: Whether to use Monte Carlo dropout
            var: Whether to estimate variance (for NLL loss)
        """
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        
        self.quantiles = quantiles
        self.monte_carlo = monte_carlo
        self.var = var
        
        # Adjust output dimension for quantile regression
        if self.quantiles is not None:
            self.output_dim = self.output_dim * len(quantiles)
        
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
        self.fc = nn.Linear(fc_input_dim, self.output_dim * self.horizon)
        if self.var:
            self.fc_logvar = nn.Linear(fc_input_dim, self.output_dim * self.horizon)

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
            out = self.fc(x)
            return out.view(-1, self.horizon, self.output_dim), var.view(-1, self.horizon, self.output_dim)
            
        if self.quantiles is not None:
            # Return reshaped output for quantile regression
            out = self.fc(x)
            return out.view(-1, self.horizon, self.output_dim // len(self.quantiles), len(self.quantiles))
        
        if self.monte_carlo:
            x = nn.Dropout(p=self.config.dropout)(x)
            out = self.fc(x)
            return out.view(-1, self.horizon, self.output_dim)
        
        # Standard or Monte Carlo dropout output
        return self.fc(x).view(-1, self.horizon, self.output_dim)

# Encoder and Decoder CNNs

# Multi-Layer Perceptrons (ANNs)

activations = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid", "Softmax", "LogSoftmax"]

class MLP(BaseModel):
    """Unified MLP implementation supporting multiple uncertainty estimation methods"""
    def __init__(self, config: MLPConfig, input_dim: int, output_dim: int,
                 horizon: int, quantiles: Optional[List[float]] = None,
                 monte_carlo: Optional[bool] = False,
                 var: Optional[bool] = False):
        """Initialize MLP model with various uncertainty estimation capabilities
        
        Args:
            config: MLPConfig object containing model parameters
            input_dim: Input dimension: Number of features x time horizon
            output_dim: Output dimension
            horizon: Prediction horizon
            quantiles: List of quantiles for quantile regression
            monte_carlo: Whether to use Monte Carlo dropout
            var: Whether to estimate variance (for NLL loss)
        """
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
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
        self.fc = nn.Linear(current_dim, self.output_dim * self.horizon)
        if self.var:
            self.fc_logvar = nn.Linear(current_dim, self.output_dim * self.horizon)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to deal with batching
        x = x.view(x.size(0), -1) # Flattens to shape of (batch, horizon)
        x = self.layers(x)
        
        if self.var:
            # Return mean and variance for NLL loss
            var = torch.exp(self.fc_logvar(x))
            x = self.fc(x)
            
            return x.view(-1, self.horizon, self.output_dim), var.view(-1, self.horizon, self.output_dim)
        
        if self.monte_carlo: # Dropout from the output
        # Pass the output 
            x = nn.Dropout(p=self.config.dropout)(x)
            out = self.fc(x)
            # shape of out is [batch_size, output_dim], reshape to [batch, horizon, features]
            return out.view(-1, self.horizon, self.output_dim)
        
        if self.quantiles is not None:
            out = self.fc(x)
            # Return reshaped output for quantile regression
            return out.view(-1, self.horizon, self.output_dim // len(self.quantiles), len(self.quantiles))
            
        # Standard or Monte Carlo dropout output
        return self.fc(x)

# Linear Regression
class MLR(BaseModel):
    """Multi Linear Regression Model"""
    def __init__(self, config: TrainingConfig, input_dim: int, output_dim: int, 
                 horizon: int, quantiles: Optional[List[float]] = None, 
                 monte_carlo: Optional[bool] = False, 
                 var: Optional[bool] = False):
        
        super().__init__(config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        
        self.quantiles = quantiles
        self.var = var
        self.monte_carlo = monte_carlo
        
        if self.quantiles is not None:
            self.output_dim = output_dim * len(self.quantiles)
        
        if self.var:
            self.logvar = nn.Linear(self.input_dim, self.output_dim * self.horizon)
        
        
        self.fc = nn.Linear(self.input_dim, self.output_dim * self.horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to deal with batching
        x = x.view(x.size(0), -1) # Flattens to shape of (batch, horizon)
        out = self.fc(x)
        
        if self.var:
            var = torch.exp(self.logvar(x))
            return out.view(-1, self.horizon, self.output_dim), var.view(-1, self.horizon, self.output_dim)
        
        if self.quantiles is not None:
            return out.view(-1, self.horizon, self.output_dim // len(self.quantiles), len(self.quantiles))
        
        if self.monte_carlo:
            out = nn.Dropout(p=self.config.dropout)(out)
            return out.view(-1, self.horizon, self.output_dim)
        
        return out
    
# Transformers
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        
        # Ensure div_term matches the even dimensions of d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

# Encoder Only Model
class EncoderOnlyTransformer(BaseModel):
    def __init__(self, config: TrainingConfig, input_dim: int, output_dim: int,
                 horizon: int, quantiles: Optional[List[float]] = None, 
                 monte_carlo: Optional[bool] = False, var: Optional[bool] = False):
        super().__init__(config)
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        
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
        
        # Output layers for using encoder only model
        self.output_layer = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dim_feedforward, self.output_dim * self.horizon)
        )
        
        if self.var:            
            # self.fc_logvar = nn.Linear(self.config.d_model, self.output_dim)
            self.fc_logvar = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.dim_feedforward),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.dim_feedforward, self.output_dim * self.horizon)
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
            return output.view(-1, self.horizon, self.output_dim), var.view(-1, self.horizon, self.output_dim)
        
        if self.monte_carlo:
            output = nn.Dropout(p=self.config.dropout)(output_seq)
            output_final = self.output_layer(output)
            return output_final.view(-1, self.horizon, self.output_dim)
        
        if self.quantiles is not None:
            return output.view(-1, self.horizon, self.output_dim // len(self.quantiles), len(self.quantiles))
        
        return output.view(-1, self.horizon, self.output_dim)

class DecoderOnlyTransformer(BaseModel):
    def __init__(self, config: TrainingConfig, input_dim: int, output_dim: int,
                 horizon: int, quantiles: Optional[List[float]] = None, 
                 monte_carlo: Optional[bool] = False, var: Optional[bool] = False):
        super().__init__(config)
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        
        self.quantiles = quantiles
        self.monte_carlo = monte_carlo
        self.var = var
        
        if self.quantiles is not None:
            self.output_dim = output_dim * len(quantiles)
        
        # Input projection layer
        self.input_projection = nn.Linear(self.input_dim, self.config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.config.d_model)
        
        # Transformer decoder layers
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True
        )
        
        # Transformer decoder (using self-attention only)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layers, 
            num_layers=self.config.num_layers
        )
        
        # Output layers for forecasting
        self.output_layer = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dim_feedforward, self.output_dim)
        )
        
        if self.var:
            self.fc_logvar = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.dim_feedforward),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.dim_feedforward, self.output_dim)
            )

    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x, target_seq=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate causal mask for self-attention
        # This ensures the model only looks at previous positions
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # If we're in inference mode and target_seq is provided
        if target_seq is not None:
            # Prepare autoregressive generation
            # Concatenate the input and target sequences
            target_seq = self.input_projection(target_seq)
            target_seq = self.pos_encoder(target_seq)
            
            combined_seq = torch.cat([x, target_seq], dim=1)
            # Update mask to account for the combined sequence length
            mask = self._generate_square_subsequent_mask(combined_seq.size(1)).to(x.device)
            
            # Apply transformer decoder
            # In decoder-only mode, we pass the same tensor as both tgt and memory
            decoder_output = self.transformer_decoder(
                tgt=combined_seq,
                memory=combined_seq,  # Same as target in decoder-only architecture
                tgt_mask=mask
            )
            
            # Extract just the forecast part of the output
            decoder_output = decoder_output[:, -self.horizon:, :]
        else:
            # Apply transformer decoder with self-attention only
            # In decoder-only mode, we pass the same tensor as both tgt and memory
            decoder_output = self.transformer_decoder(
                tgt=x,
                memory=x,  # Same as target in decoder-only architecture
                tgt_mask=mask
            )
            
            # Take the last 'horizon' elements for prediction
            decoder_output = decoder_output[:, -self.horizon:, :]
        
        # Create outputs for each timestep in the horizon
        forecasts = []
        variances = [] if self.var else None
        
        for i in range(self.horizon):
            # Apply output layer to each position
            step_output = self.output_layer(decoder_output[:, i])
            forecasts.append(step_output)
            
            # If variance is needed, compute it for each step
            if self.var:
                step_var = torch.exp(self.fc_logvar(decoder_output[:, i]))
                variances.append(step_var)
        
        # Stack outputs along a new dimension to get [batch, horizon, features]
        forecasts = torch.stack(forecasts, dim=1)
        
        if self.var:
            variances = torch.stack(variances, dim=1)
            return forecasts, variances
        
        if self.monte_carlo: # Dropout from the output
        # Pass the output 
            x = nn.Dropout(p=self.config.dropout)(x)
            out = self.fc(x)
            # shape of out is [batch_size, output_dim], reshape to [batch, horizon, features]
            return out.view(-1, self.horizon, self.output_dim)
        
        if self.quantiles is not None:
            # Reshape for quantile outputs
            return forecasts.view(batch_size, self.horizon, self.output_dim // len(self.quantiles), len(self.quantiles))
        
        return forecasts


# Encoder and Decoder Transformers

class EncoderTransformer(BaseModel):
    """Transformer Encoder model"""
    def __init__(self, config: TFConfig, input_dim: int,
                 monte_carlo: Optional[bool] = False):
        super().__init__(config)
        
        self.input_dim = input_dim
        self.monte_carlo = monte_carlo
        
        # Input normalization and dropout
        self.input_bn = nn.BatchNorm1d(input_dim) if getattr(self.config.norm_type, 'batch', False) else None
        self.input_ln = nn.LayerNorm(input_dim) if getattr(self.config.norm_type, 'layer', False) else None
        self.input_do = nn.Dropout(p=self.config.dropout) if monte_carlo else None
        
        # Normalization for the encoder
        self.ec_norm = nn.BatchNorm1d(self.config.d_model) if getattr(self.config.norm_type, 'batch', False) else nn.LayerNorm(self.config.d_model) if getattr(self.config.norm_type, 'layer', False) else None
        
        # Input embedding layer
        self.encoder_input_layer = nn.Linear(
            in_features=input_dim, 
            out_features=self.config.d_model 
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.config.d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model, 
            nhead=self.config.num_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True
        )
        
        # Stack encoder layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.config.num_layers, 
            norm=self.ec_norm
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Apply normalization and dropout to input if specified
        if self.input_bn is not None:
            x = x.transpose(1, 2)
            x = self.input_bn(x)
            x = x.transpose(1, 2)
        if self.input_ln is not None:
            x = self.input_ln(x)
        if self.input_do is not None:
            x = self.input_do(x)
        
        # Input embedding
        x = self.encoder_input_layer(x)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Pass through encoder
        encoder_output = self.encoder(x, src_key_padding_mask=mask)
        
        return encoder_output
    
class DecoderTransformer(BaseModel):
    """Transformer Decoder model"""
    def __init__(self, config: TFConfig, hidden_dim: int, output_dim: int, horizon: int,
                 quantiles: Optional[List[float]] = None,
                 monte_carlo: Optional[bool] = False, 
                 var: Optional[bool] = False):
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.quantiles = quantiles
        self.monte_carlo = monte_carlo
        self.var = var
        
        # Adjust output dimension for quantile regression if specified
        if self.quantiles:
            self.output_dim = self.output_dim * len(self.quantiles)
        
        # Normalization for the decoder
        self.dec_norm = nn.BatchNorm1d(self.config.d_model) if getattr(self.config.norm_type, 'batch', False) else nn.LayerNorm(self.config.d_model) if getattr(self.config.norm_type, 'layer', False) else None
        
        # Input layer for decoder
        self.decoder_input_layer = nn.Linear(
            in_features=hidden_dim,
            out_features=self.config.d_model
        )
        
        # Output mapping layer
        self.linear_mapping = nn.Linear(
            in_features=self.config.d_model, 
            out_features=self.output_dim
        )
        
        # Variance layer for probabilistic forecasting
        if self.var:
            self.var_mapping = nn.Linear(
                in_features=self.config.d_model,
                out_features=self.output_dim
            )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True
        )
        
        # Stack decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.config.num_layers, 
            norm=self.dec_norm
        )
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None) -> torch.Tensor:
        # Process the decoder input sequence
        decoder_input = self.decoder_input_layer(tgt)
        
        # Pass through transformer decoder
        decoder_output = self.decoder(
            decoder_input,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        
        # Handle variance for probabilistic forecasting
        if self.var:
            out = self.linear_mapping(decoder_output)
            var = torch.exp(self.var_mapping(decoder_output))
            return out, var
        
        # Handle quantile regression
        if self.quantiles is not None:
            out = self.linear_mapping(decoder_output)
            batch_size = out.size(0)
            return out.view(batch_size, self.horizon, self.output_dim // len(self.quantiles), len(self.quantiles))
        
        # Standard output
        out = self.linear_mapping(decoder_output)
        batch_size = out.size(0)
        return out.view(batch_size, self.horizon, -1)
    
class EncoderDecoderTransformer(BaseEncoderDecoder):
    """Complete Encoder-Decoder Transformer model"""
    def __init__(self,
        encoder_config: TFConfig,
        decoder_config: TFConfig,
        input_dim: int,
        horizon: int,
        output_dim: int,
        var: Optional[bool] = None,
        monte_carlo: Optional[bool] = None,
        quantiles: Optional[List] = None,
        ):
        """
        Args:
            encoder_config: Configuration for the encoder
            decoder_config: Configuration for the decoder
            input_dim: Number of input variables
            horizon: Forecast horizon
            output_dim: Number of output variables
            var: Whether to output variance for probabilistic forecasting
            monte_carlo: Whether to use Monte Carlo dropout for uncertainty estimation
            quantiles: List of quantiles for quantile regression
        """
        super().__init__(encoder_config, decoder_config)
        
        self.input_dim = input_dim
        self.horizon = horizon
        self.output_dim = output_dim
        self.var = var
        self.monte_carlo = monte_carlo
        self.quantiles = quantiles
        
        # Adjust output dimension for quantile regression if specified
        if self.quantiles:
            self.output_dim = self.output_dim * len(self.quantiles)
        
        # Create the encoder
        self.encoder_model = EncoderTransformer(
            config=encoder_config,
            input_dim=input_dim,
            monte_carlo=monte_carlo
        )
        
        # Create the decoder
        self.decoder_model = DecoderTransformer(
            config=decoder_config,
            hidden_dim=encoder_config.d_model,
            output_dim=output_dim,
            horizon=horizon,
            quantiles=quantiles,
            monte_carlo=monte_carlo,
            var=var
        )
    
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape [batch_size, seq_len, input_dim]
            x_mask: Mask for the encoder inputs
            tgt_mask: Mask for the decoder inputs
        """
        # Get encoder outputs
        encoder_output = self.encoder_model(x, mask=x_mask)
        
        # For transformer models, we need to initialize a decoder input sequence
        # We'll use the last time step of the encoder output to initialize
        batch_size = x.size(0)
        
        # Create a target sequence of the appropriate length (horizon)
        # Initially, we'll use the last time step of encoder output repeated horizon times
        decoder_input = encoder_output[:, -1:, :].repeat(1, self.horizon, 1)
        
        # Use a causal mask for autoregressive generation if tgt_mask is not provided
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                self.horizon).to(x.device)
        
        # Pass encoder outputs and decoder input to decoder
        decoder_output = self.decoder_model(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            memory_mask=None
        )
        
        # Reshape output to [batch_size, horizon, output_dim]
        if self.monte_carlo: # Dropout from the output
        # Pass the output 
            x = nn.Dropout(p=self.config.dropout)(decoder_output)
            decoder_output = self.fc(x)
            
        if self.var:
            out, var = decoder_output
            return out.view(batch_size, self.horizon, -1), var.view(batch_size, self.horizon, -1)
        elif self.quantiles:
            return decoder_output.view(batch_size, self.horizon, -1, len(self.quantiles))
        else:
            return decoder_output.view(batch_size, self.horizon, -1)
        

# Graph Neural Networks (GNNs)

class ST_GCN(BaseModel):
    """Spatial-Temporal Graph Convolutional Network"""
    