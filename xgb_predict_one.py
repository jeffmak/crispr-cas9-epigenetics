import pandas as pd
import xgboost as xgb

# This Python script uses the following data and model files:
# 1. An example input CSV file with one row, namely one CRISPR-Cas9 cleavage
#    activity datapoint from the crisprSQL dataset in the paper
example_loc = 'data/example_input.csv'
# 2. The XGBoost tree model used for making CRISPR-Cas9 cleavage activity
#    prediction
xgb_loc = 'models/xgb_model.json'

# Create and load the XGBoost model
# Use XGBRegressor since CRISPR-Cas9 cleavage activity values are continuous
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(xgb_loc)

# Read the input CSV file
example_df = pd.read_csv(example_loc)

# Make the prediction
pred = xgb_model.predict(example_df.astype('float'))

# Print the result
# -1.054 is the predicted CRISPR-Cas9 cleavage activity value
print('Predicted CRISPR-Cas9 Cleavage Activity: {:.3f}'.format(pred[0]))
