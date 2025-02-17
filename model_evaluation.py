# Want to evaluate based on
# 1. Accuracy - MSE, RMSE etc.
# 2. How well are uncertainties covered - through coverage ratio and MLE estimation (quantile?)
# 3. The complexity of the model - number of parameters and FLOPs - can report training time if really want to

from utils import ModelEvaluation

model_path = 'models/'


evaluation = ModelEvaluation()