import torch
from models import *
from utils import *
from simulation import *



path = "models/CSTR_NLL_<class 'models.MLR'>_model_7.pth"

model_type = MLR

cp = torch.load(path)
print("Training configuration: ", cp['training_config'])
print("Model parameters: ", cp['model_config'])
model = model_type(cp['model_config'],
                input_dim=X_train.shape[1] * X_train.shape[2], # check this for MLR/LSTM etc.
                output_dim=y_train.shape[2],
                horizon = cp['training_config'].horizon,
                var = True
            )
model.load_state_dict(cp['model_state_dict'])
# Then find the complexity of the model
flops = FlopCountAnalysis(model, X_train.to(training_config.device))
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

sequence_length = cp['training_config'].time_step
horizon = cp['training_config'].horizon
feature_names = ['Concentration', 'Temperature']
visualiser = Visualizer()

visualiser.plot_predictions(means, features,
                            noiseless_results,
                            sequence_length,
                            horizon, feature_names,
                            num_simulations = 10,
                            train_test_split = 0.8,
                            vars = variances)