import torch
from models import *
from utils import *
import pandas as pd

noiseless = pd.read_csv('datasets/cstr_noiseless_results.csv')
features = pd.read_csv('datasets/cstr_sim_features.csv')
targets = pd.read_csv('datasets/cstr_sim_targets.csv')


path = "/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/BenchmarkTSA/models/CSTR_NLL_<class 'models.LSTM'>_model_45.pth"

model_type = LSTM

cp = torch.load(path)
print("Training configuration: ", cp['training_config'])
print("Model parameters: ", cp['model_config'])

data_processor = DataProcessor(cp['training_config'], features, targets)
(train_loader, test_loader, val_loader, X_train, X_test, X_val, y_train, y_test, y_val, X, y) = data_processor.prepare_data()

model = model_type(cp['model_config'],
                input_dim=X_train.shape[2], # check this for MLR/LSTM etc.
                output_dim=y_train.shape[2],
                horizon = cp['training_config'].horizon,
                var = True
            )
model.load_state_dict(cp['model_state_dict'])
# Then find the complexity of the model
flops = FlopCountAnalysis(model, X_train.to(cp['training_config'].device))
print(f"Number of FLOPs: {flops.total()}")
print(f"Number of Model Params: {sum(p.numel() for p in model.parameters())}")

model.eval()



with torch.no_grad():
    mean, var = model(X)


mean = mean.detach().numpy()
mean = data_processor.reconstruct_sequence(mean, True)
var = var.detach().numpy()
var = data_processor.reconstruct_sequence(var, True)

rescaled_pred = data_processor.rescale_predictions(mean, var)
means = rescaled_pred[0]
variances = rescaled_pred[1]
print(variances.shape)

sequence_length = cp['training_config'].time_step
horizon = cp['training_config'].horizon
feature_names = ['Concentration', 'Temperature']
visualiser = Visualizer()

visualiser.plot_preds(means, features,
                            noiseless_results,
                            sequence_length,
                            horizon, feature_names,
                            num_simulations = 10,
                            train_test_split = 0.6, test_val_split = 0.8,
                            vars = variances)