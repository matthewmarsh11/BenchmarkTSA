# Import dependencies

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import norm
from tqdm import tqdm
from base import TrainingConfig, BaseModel, CNNConfig, LSTMConfig
from models import *
from Bioprocess_Sim import *
from CSTR_Sim import *
from utils import *
np.random.seed(42)
torch.manual_seed(42)

features_path = 'datasets/cstr_sim_features.csv'
targets_path = 'datasets/cstr_sim_targets.csv'
noiseless = pd.read_csv('datasets/cstr_noiseless_results.csv')
features = pd.read_csv(features_path)
targets = pd.read_csv(targets_path)

# Initial model and training configurations

training_config = TrainingConfig(
    batch_size = 30,
    num_epochs = 100,
    learning_rate = 0.001,
    time_step = 15,
    horizon = 5,
    weight_decay = 0.01,
    factor = 0.8,
    patience = 10,
    delta = 0.1,
    train_test_split = 0.6,
    test_val_split = 0.8,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
)

training_bounds = {
    
        'batch_size': (2, len(features)),
        'num_epochs': (50, 1500),
        'learning_rate': (0.0001, 0.1),
        'time_step': (2, 20),
        'horizon': (3, 10),
        'weight_decay': (1e-6, 0.1),
        'factor': (0.1, 0.99),
        'patience': (5, 100),
        'delta': (1e-6, 0.1),   
}

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

LSTM_bounds = {
    'hidden_dim': (32, 1024),
    'num_layers': (1, 50),
    'dropout': (0.1, 0.9),
    'bidirectional': (0, 1),
    'norm_type': (0, 2),
}

CNN_bounds = {
    'conv_channels': [(4, 512), (8, 1024)],   # Allowing very small and extremely large channel sizes
    'kernel_sizes': [(1, 15), (1, 11)],       # Very small (1) to very large (15) receptive fields
    'fc_dims': [(16, 2048), (32, 4096)],      # Fully connected layers from tiny to massive
    'dropout': (0.0, 0.9),                    # Almost full range of dropout (up to extreme regularization)
}

TF_bounds = {
    'num_layers': (1, 50),
    'hidden_dim': (32, 1024),
    'd_model': (32, 1024),
    'num_heads': (1, 8),
    'dim_feedforward': (32, 1024),
    'dropout': (0.1, 0.9),
}

MLP_bounds = {
    'hidden_dim': (32, 1024),
    'num_layers': (1, 50),
    'dropout': (0.1, 0.9),
    'activation': (0, 7),  # Assuming 0: ReLU, 1: Tanh, 2: Sigmoid
}

MLR_bounds = {
    'dropout': (0.1, 0.9),
}

# Define bounds for parameters


model_dict = {
    LSTM : (LSTM_Config, {**training_bounds, **LSTM_bounds}),
    EncoderDecoderLSTM: (LSTM_Config, {**training_bounds, **LSTM_bounds}),
    CNN: (CNN_Config, {**training_bounds, **CNN_bounds}),
    EncoderOnlyTransformer: (TF_Config, {**training_bounds, **TF_bounds}),
    DecoderOnlyTransformer: (TF_Config, {**training_bounds, **TF_bounds}),
    EncoderDecoderTransformer: (TF_Config, {**training_bounds, **TF_bounds}),
    MLP: (MLP_Config, {**training_bounds, **MLP_bounds}),
    MLR: (MLR_Config, {**training_bounds, **MLR_bounds})
}

# Iterate through each defined model to optimise

model = EncoderDecoderLSTM
model_config = model_dict[model][0]
bounds = model_dict[model][1]

# Define the path to save the model to

quantiles = [0.1, 0.5, 0.9]

optimiser = ModelOptimisation(
    model_class= model,
    train_config = training_config,
    model_config = model_config,
    config_bounds=bounds,
    features_path = features_path,
    targets_path = targets_path,
    converter = CSTRConverter,
    data_processor = DataProcessor,
    trainer_class = ModelTrainer,
    iters = 200,
    quantiles = None, # Define how to quantify uncertainty
    monte_carlo = None,
    variance = True,
    
)

if optimiser.quantiles:
    uncertainty = 'qtle'

if optimiser.monte_carlo:
    uncertainty = 'MC'

if optimiser.variance:
    uncertainty = 'NLL'


path = f'models/CSTR_{uncertainty}_{model}.pth'

best_params, best_loss  = optimiser.optimise(path)