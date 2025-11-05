# MSPC

The general workflow is to:

1. Run the system with a nominal setpoint and controller to collect data with the data logger.
2. Align and preprocess logged data
3. Train a predictive model + controller
4. Deploy the predictive controller at new system operation runtime using MSPC controller type.

The XXX_alignment notebooks are for data preprocessing prior to training.

prediction.ipynb is the main training and eval script

mspc.py defines the predictors and controllers
