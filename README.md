# Test Task 2

The following repository contains solution for second task. The solution uses xgboost model to predict target value. Repository consists of the following files:
- `eda.ipynb` jupyter notebook with exploratory data analysis;
- `train.py` python script for model training;
- `predict.py` python script for model inference on test data;
- file with prediction results;
- `requirements.txt` file.

Originally, `eda.ipynb` was too big to upload to GitHub, so I removed all outputs. Please, run it to see visualizations, in the same directory as `train.csv`.
To install all the required packages run following command:

`pip install -r requirements.txt`

Use the following command to train with default parameters:

`python train.py --data_path path/to/train.csv --model_path xgboost_model.joblib`

By default, hyperparameters found in EDA will be used when training model, but it is possible to specify other values via command line.

After training, use this command to predict:

`python predict.py --data_path path/to/hidden_test.csv --model_path xgboost_model.joblib --output_path predictions.csv`

