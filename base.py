from dataclasses import dataclass
import torch.nn as nn
from abc import ABC, abstractmethod
import torch

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    batch_size: int
    num_epochs: int
    learning_rate: float
    time_step: int
    horizon: int
    weight_decay: float
    factor: float
    patience: int
    delta: float
    train_test_split: float = 0.6
    test_val_split: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class LSTMConfig:
    """Configuration for LSTM model"""
    num_layers: int
    hidden_dim: int
    dropout: float = 0.2
    bidirectional: bool = False
    norm_type: str = 'batch'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class CNNConfig:
    """Configuration for CNN model"""
    conv_channels: int
    kernel_sizes: int
    fc_dims: int
    dropout: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class MLPConfig:
    """Configuration for MLP model"""
    hidden_dim: int
    num_layers: int
    dropout: float = 0.2
    activation: str = 'ReLU'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TFConfig:
    """Configuration for Transformer model"""
    num_layers: int
    hidden_dim: int
    d_model: int
    num_heads: int
    dim_feedforward: int
    dropout: float = 0.2
    norm_type: str = 'batch'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    

@dataclass
class MLRConfig:
    """Configuration for MLR model"""
    dropout: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models"""
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
class BaseEncoderDecoder(nn.Module, ABC):
    """Abstract base class for all encoder-decoder models"""
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass