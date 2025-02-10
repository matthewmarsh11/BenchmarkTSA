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
from tqdm.notebook import tqdm
np.random.seed(42)

# Create simulation

simulation_config = SimulationConfig(
    n_simulations=100,
    T = 100, # Change the number of time steps
    tsim = 500,
    noise_percentage = 0.01
    )

simulator = CSTRSimulator(simulation_config) # Change this for different case studies
simulation_results, noiseless_simulation = simulator.run_multiple_simulations()

# Convert simulation results to feed into model

converter = CSTRConverter() # Change this for different case studies
features, targets = converter.convert(simulation_results)
noiseless_results, _ = converter.convert(noiseless_simulation) # To compare model to noiseless simulation

# Initial model and training configurations

training_config = TrainingConfig(
    batch_size = 101,
    num_epochs = 1000,
    learning_rate = 0.001,
    time_step = 15,
    horizon = 5,
    weight_decay = 0.01,
    factor = 0.8,
    patience = 10,
    delta = 0.1,
    train_test_split = 0.8,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
)

model_config = LSTMConfig(
    hidden_dim = 64,
    num_layers=4,
    dropout = 0.2,
    bidirectional=False,
    norm_type = None,
)

# Define bounds for parameters

training_bounds = {
    
        'batch_size': (2, len(features)) if isinstance(simulator, CSTRSimulator) else (2, 10),
        'num_epochs': (50, 1500),
        'learning_rate': (0.0001, 0.1),
        'time_step': (2, 20) if isinstance(simulator, CSTRSimulator) else (2, 10),
        'horizon': (3, 10),
        'weight_decay': (1e-6, 0.1),
        'factor': (0.1, 0.99),
        'patience': (5, 100),
        'delta': (1e-6, 0.1),   
}

# Change these bounds depending on model chosen
hyperparameter_bounds = {
        'hidden_dim': (32, 512),
        'num_layers': (1, 20),
        'dropout': (0.1, 0.9),
        'bidirectional': (0, 1),
        'norm_type': (0, 2),
    
}

# Combine training and hyperparameter bounds
bounds = {**training_bounds, **hyperparameter_bounds}

# Define the path you want to save the model to
path = 'models/CSTR_NLL_LSTM.pth'

# Define the model
model_class = LSTM

optimiser = ModelOptimisation(
    model_class= model_class,
    sim_config = simulation_config,
    train_config = training_config,
    model_config = model_config,
    config_bounds=bounds,
    simulator = CSTRSimulator,
    converter = CSTRConverter,
    data_processor = DataProcessor,
    trainer_class = ModelTrainer,
    iters = 300,
    quantiles = None, # Define how to quantify uncertainty
    monte_carlo = None,
    variance = True,
    
)

best_params, best_los  = optimiser.optimise(path)