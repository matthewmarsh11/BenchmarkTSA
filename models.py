import torch
import torch.nn as nn
from base import TrainingConfig, BaseModel
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass


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
        self.output_dim = output_dim * len(quantiles)
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

