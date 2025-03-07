from Bioprocess_Sim import *
from CSTR_Sim import *
from utils import *
import pandas as pd

# Simulate the CSTR 10 times, with 5000 timesteps over 1000 second period
CSTR_Config = SimulationConfig(n_simulations=100,
                                T = 101,
                                tsim = 500,
                                noise_percentage=0.01,
                            ) 

BP_Config = SimulationConfig(n_simulations=10,
                                T = 20,
                                tsim = 240,
                                noise_percentage=0.01,
)

# simulator = CSTRSimulator(CSTR_Config)
simulator = CSTRSimulator(CSTR_Config)

simulation_results, noiseless_sim = simulator.run_multiple_simulations()

# Plot the output of the Simulation
# simulator.plot_results(simulation_results, noiseless_sim)
# converter = BioprocessConverter()
converter = CSTRConverter()
features, targets = converter.convert(simulation_results)
noiseless_results, _ = converter.convert(noiseless_sim)

# Save the features and targets to CSV files
features_df = pd.DataFrame(features)
targets_df = pd.DataFrame(targets)
noiseless_results_df = pd.DataFrame(noiseless_results)

features_df.to_csv('/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/BenchmarkTSA/datasets/cstr/small_cstr_features.csv', index=False)
targets_df.to_csv('/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/BenchmarkTSA/datasets/cstr/small_cstr_targets.csv', index=False)
noiseless_results_df.to_csv('/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/BenchmarkTSA/datasets/cstr/small_cstr_noiseless_results.csv', index=False)
# # Define a preliminary training configuration for the model
# # Data processing uses an initial lookback region of 5 timesteps to predict 1 in the future 
# # with an 80% train test split and a batch size of 4
# training_config = TrainingConfig(
#     batch_size = 4,
#     num_epochs = 50,
#     learning_rate = 0.001,
#     time_step = 10,
#     horizon = 5,
#     weight_decay = 0.01,
#     factor = 0.8,
#     patience = 10,
#     delta = 0.1,
#     train_test_split = 0.8,
#     device = 'cuda' if torch.cuda.is_available() else 'cpu',
# )

# data_processor = DataProcessor(training_config, features, targets)
# (train_loader, test_loader, val_loader, X_train, X_test, X_val, y_train, y_test, y_val, X, y) = data_processor.prepare_data()