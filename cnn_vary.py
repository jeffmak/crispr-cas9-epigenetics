import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from cnn import ConvolutionalNet, vecToMatEncoding

# Nucleotide BDM, GC147 and NuPoP (Affinity) are identified as important
# computed nucleosome organization-related scores. Thus, in this Python script,
# we vary the values of these scores, and see how such varied values
# affect XGBoost's predictions.

# This Python script uses the following data and model files:
# 1. An input CSV file containing 2000 random experimental/augmented datapoints
#    from the crisprSQL dataset in the paper
csv_loc = 'data/crisprSQL_dataset_2000.csv'
# 2. PyTorch model file containing the convolutional neural network (CNN) model
#    which predicts CRISPR-Cas9 (off-)target cleavage activity.
state_dict_loc = 'models/cnn_model.pt'

# The plots generated will be saved in these relative locations
out_dict = {'Nucleotide BDM': 'out/cnn_vary_nucleotide_bdm.pdf',
            'GC147': 'out/cnn_vary_gc147.pdf',
            'NuPoP (Affinity)': 'out/cnn_vary_nupop_affinity.pdf'}

### IMPORTANT CONSTANTS ###
# Don't change these!
TAR_SEQ_LEN = 23 # target DNA sequence's length
epiDim = 22 # The total number of epigenetic features considered
numBpWise = 13 # Number of base pair-resolved nucleosome organization-related
               # scores/features

# Feel free to change
NUM_SAMPLES = 5 # Number of random datapoints to select for visualization,
                 # i.e., number of row to select from the input CSV file
NUM_VARIED = 20 # Number of varied points per random datapoint
###########################

# Read the input CSV file
X_df = pd.read_csv(csv_loc)

# Decide which compute device to use when using PyTorch.
# Use a GPU if available, CPU otherwise.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create convolutional neural network regression PyTorch model
torch_model = ConvolutionalNet(epiDim, device=device)

# Load saved PyTorch weights
state_dict = torch.load(state_dict_loc, map_location=device)
torch_model.load_state_dict(state_dict)

# These are the column/feature names corresponding to ...
# (a) Nucleotide BDM
NBDM_feats = ['NucleotideBDM_' + str(i) for i in range(1, 1 + TAR_SEQ_LEN)]
# (b) GC147
GC_feats = ['GCContent_' + str(i) for i in range(1, 1 + TAR_SEQ_LEN)]
# (c) NuPoP (Affinity)
NuPoP_Aff_feats = ['NuPoP_Affinity_147_h'] + \
                  ['NuPoP_Affinity_147_h.' + str(i) for i in range(1,
                                                                  TAR_SEQ_LEN)]
# Put these features into a dictionary
feats_dict = {'Nucleotide BDM': NBDM_feats,
              'GC147': GC_feats,
              'NuPoP (Affinity)': NuPoP_Aff_feats}

# To obtain sensible value ranges, we exclude data points with zeros across
# all computed nucleosome organization-related score, since these zeros arise
# from data imputation during training.
non_zero_X_df = X_df[X_df['GCContent_1'] > 0]

# Set values ranges for each computed score to be within the minimum and
# maximum value of the respective features, i.e., we vary within the ranges
# of the dataset.
ranges_dict = {'Nucleotide BDM': (non_zero_X_df[NBDM_feats].min(axis=1).min(),
                                  non_zero_X_df[NBDM_feats].max(axis=1).max()),
              'GC147': (non_zero_X_df[GC_feats].min(axis=1).min(),
                        non_zero_X_df[GC_feats].max(axis=1).max()),
              'NuPoP (Affinity)': (non_zero_X_df[NuPoP_Aff_feats].min(axis=1).min(),
                                   non_zero_X_df[NuPoP_Aff_feats].max(axis=1).max())}

# We select five random data points...
X_sample = X_df.sample(NUM_SAMPLES)

# Generate plots for each of the three computed scores
for feat_name, feats in feats_dict.items():
  min_val, max_val = ranges_dict[feat_name]
  print("{}'s value ranges from {:.2f} to {:.2f}".format(feat_name,
                                                         min_val, max_val))

  # For each data point...
  for i in range(NUM_SAMPLES):
    point_chosen = X_sample.iloc[i]

    # Create a Pandas dataframe for the random datapoint
    example_df = pd.DataFrame(point_chosen).transpose()

    # Determine the varied values to change to
    varied_vals = np.linspace(min_val, max_val, NUM_VARIED)

    # Create the artificial datapoints containing the varied values
    vary_df = example_df.loc[example_df.index.repeat(len(varied_vals))]
    for feat in feats:
      vary_df[feat] = varied_vals

    # # Make predictions for these artificial datapoints
    # preds = xgb_model.predict(vary_df)

    # Convert the Pandas DataFrame into a PyTorch tensor
    X_torch = torch.as_tensor(vary_df.to_numpy()).float()

    # Format the PyTorch tensor to fit the CNN's input dimensions
    x = vecToMatEncoding(X_torch, numBpWise=13)

    # Prepare for model inference by setting torch_model to evaluation mode
    torch_model.eval()

    # Make CRISPR-Cas9 cleavage activity predictions for these artificial datapoints
    preds = torch_model(x).cpu().flatten().detach().numpy()

    # Plot the line
    plt.plot(varied_vals, preds)

  # Set...
  # the plot's minimum and maximum x-value, ...
  plt.xlim(min_val, max_val)
  # the x-axis's name, and
  plt.xlabel("Varied {} Value".format(feat_name))
  # the y-axis's name
  plt.ylabel("Predicted CRISPR-Cas9 Cleavage Activity")

  # This is not a Jupyter notebook, so we won't show the figure here
  # plt.show()

  # Save the plot
  plt.savefig(out_dict[feat_name])

  # Close the current figure
  plt.close()

# The script prints:
# Nucleotide BDM's value ranges from 172.66 to 480.48
# GC147's value ranges from 0.25 to 0.86
# NuPoP (Affinity)'s value ranges from 0.50 to 2.13

# We observe the following tendencies in the plots:
#   - increasing Nucleotide BDM value increases predicted CRISPR-Cas9 cleavage
#     activity value
#   - increasing GC147 increases predicted CRISPR-Cas9 cleavage activity value
#   - increasing NuPoP (Affinity) value decreases predicted CRISPR-Cas9 cleavage
#     activity value
