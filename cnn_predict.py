import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from cnn import ConvolutionalNet, vecToMatEncoding

# This Python script uses the following data and model files:
# 1. An input CSV file containing 2000 random experimental/augmented datapoints
#    from the crisprSQL dataset in the paper
input_feat_loc = 'data/crisprSQL_dataset_2000.csv'
# 2. PyTorch model file containing the convolutional neural network (CNN) model
#    which predicts CRISPR-Cas9 (off-)target cleavage activity.
state_dict_loc = 'models/cnn_model.pt'

# We will save the predictions in the "out" folder
out_loc = 'out/cnn_preds.csv'

### IMPORTANT CONSTANTS ###
numBpWise = 13 # Number of base pair-resolved computed
               # nucleosome organization-related scores/features
epiDim = 22 # The total number of epigenetic features considered
###########################

# Read in the crisprSQL data points
X_df = pd.read_csv(input_feat_loc)

# Convert the Pandas DataFrame into a PyTorch tensor
X_torch = torch.as_tensor(X_df.to_numpy()).float()

# Decide which compute device to use when using PyTorch.
# Use a GPU if available, CPU otherwise.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create convolutional neural network regression PyTorch model
torch_model = ConvolutionalNet(epiDim, device=device)

# Load saved PyTorch weights
state_dict = torch.load(state_dict_loc, map_location=device)
torch_model.load_state_dict(state_dict)

# Format the PyTorch tensor to fit the CNN's input dimensions
x = vecToMatEncoding(X_torch, numBpWise=numBpWise)

# Prepare for model inference by setting torch_model to evaluation mode
torch_model.eval()

# Predict CRISPR-Cas9 cleavage activities
preds_y = torch_model(x).cpu().flatten().detach().numpy()

# Save the predictions
pd.DataFrame(preds_y).to_csv(out_loc)

# Print results
print('Predicted CRISPR-Cas9 Cleavage Activities:', preds_y)
# Result varies due to floating point precision arithmetic
# Example outputs:
# on CPU: [-0.669225   -4.1682196  -1.7856159  ... -0.58595717 -0.97370046
# -4.1463737 ]
# on GPU: [-0.6692251  -4.1682186  -1.7856162  ... -0.58595747 -0.9737008
# -4.1463733 ]
