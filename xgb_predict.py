import pandas as pd
import xgboost as xgb

# This Python script uses the following data and model files:
# 1. An input CSV file containing 2000 random experimental/augmented
#    CRISPR-Cas9 cleavage activity datapoints from the crisprSQL dataset
#    in the paper
csv_loc = 'data/crisprSQL_dataset_2000.csv'
# 2. The XGBoost tree model used for making CRISPR-Cas9 cleavage activity
#    prediction
xgb_loc = 'models/xgb_model.json'

# We will save the predictions in the "out" folder
out_loc = 'out/xgb_preds.csv'

# Create and load the XGBoost model
# Use XGBRegressor since CRISPR-Cas9 cleavage activity values are continuous
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(xgb_loc)

# Read the input CSV file
X_df = pd.read_csv(csv_loc)

# Make predictions
preds = xgb_model.predict(X_df)

# Save the predictions
pd.DataFrame(preds).to_csv(out_loc, index=None)

# Print results
# Predicted CRISPR-Cas9 Cleavage Activities: [-1.0535808 -3.4915433 -1.7820611 ...
# -0.7105575 -1.2672865 -4.081657 ]
print('Predicted CRISPR-Cas9 Cleavage Activities: ' + str(preds))
print('Read the full list of predictions at ' + out_loc)
