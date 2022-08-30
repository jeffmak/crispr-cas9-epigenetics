import xgboost as xgb
import pandas as pd
import xgboost as xgb
import os

dirname = os.path.dirname(__file__)

# This Python script uses the following files:

# 1. The CSV file which we read the input from
example_loc = os.path.join(dirname, 'data/example_input.csv')

# 2. The XGBoost model
xgb_loc = os.path.join(dirname, 'models/xgb_noseq_engnucepi_model_post_xgb1.6.json')


# Make
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(xgb_loc)
example_df = pd.read_csv(example_loc)
pred = xgb_model.predict(example_df.astype('float'))

# -3.440 is the predicted CRISPR-Cas9 activity value
print('Predicted CRISPR-Cas9 Activity: {:.3f}'.format(pred[0]))
