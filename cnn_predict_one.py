import pandas as pd
import torch
from cnn import ConvolutionalNet, vecToMatEncoding

# This Python script uses the following data and model files:
# 1. An example input CSV file with one row, namely one datapoint
# from the crisprSQL dataset in the paper
example_loc = 'data/example_input.csv'
# 2. PyTorch model file containing the convolutional neural network (CNN) model
#    which predicts CRISPR-Cas9 (off-)target cleavage activity.
state_dict_loc = 'models/cnn_model.pt'

### IMPORTANT CONSTANTS ###
numBpWise = 13 # Number of base pair-resolved computed
               # nucleosome organization-related scores/features
epiDim = 22 # The total number of epigenetic features considered
###########################

# Read in the crisprSQL data points
X_df = pd.read_csv(example_loc)

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

# Predict the CRISPR-Cas9 cleavage activity
pred = torch_model(x).cpu().flatten().detach().numpy()

# Print results
print('Predicted CRISPR-Cas9 Cleavage Activity: {:.3f}'.format(pred[0]))
# on CPU/GPU: -0.669
