from nn_pred import NeuralNetPredictor
import pandas as pd
import numpy as np

# This Python script uses the following data and model files:
# 1. CSV file containing five example off-target sites
input_loc = 'data/example_offtargets.csv'
# 2. PyTorch model file containing the weights for the LeNup (H3Q85C) model
model_weights_loc = 'model/lenup_h3q85c.th'

# We will save the predictions in the "data" folder
output_loc = 'data/output.csv'

# Read the input CSV file
off_data = pd.read_csv(input_loc)

# Make base pair-resolved nucleosome occupancy predictions
off_data['LeNup_H3Q85C'] = NeuralNetPredictor.batch_occupancy_scores(
                                off_data['target_context'].tolist(),
                                model_weights=model_weights_loc)
off_data['LeNup_H3Q85C'] = off_data['LeNup_H3Q85C'].apply(lambda arr: np.array(arr))

# Print the results to the console
for index, row in off_data.iterrows():
    print('Predictions for id ' + str(row['id']) + ':')
    print(np.around(row['LeNup_H3Q85C'], decimals=2))

# Save predictions to output CSV file
off_data.to_csv(output_loc)
print('Saved predictions at ' + output_loc)
