import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# In the paper, we identified Nucleotide BDM, GC147 and NuPoP (Affinity) as
# important computed nucleosome organization-related scores. Thus, in this
# Python script, we vary the values of these scores, and see how such varied
# values affect XGBoost's predictions.

# This Python script uses the following data and model files:
# 1. An input CSV file containing 2000 random experimental/augmented datapoints
#    from the crisprSQL dataset in the paper
csv_loc = 'data/crisprSQL_dataset_2000.csv'
# 2. The XGBoost tree model used for making CRISPR-Cas9 cleavage activity
#    prediction
xgb_loc = 'models/xgb_model.json'

# The plots generated will be saved in these relative locations
out_dict = {'Nucleotide BDM': 'out/xgb_vary_nucleotide_bdm.pdf',
            'GC147': 'out/xgb_vary_gc147.pdf',
            'NuPoP (Affinity)': 'out/xgb_vary_nupop_affinity.pdf'}

### IMPORTANT CONSTANTS ###
# Don't change these!
TAR_SEQ_LEN = 23 # target DNA sequence's length

# Feel free to change these
NUM_SAMPLES = 5 # Number of random datapoints to select for visualization,
                 # i.e., number of row to select from the input CSV file
NUM_VARIED = 20 # Number of varied points per random datapoint
###########################

# Read the input CSV file
X_df = pd.read_csv(csv_loc)

# Create and load the XGBoost model
# Use XGBRegressor since CRISPR-Cas9 cleavage activity values are continuous
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(xgb_loc)

# These are the column/feature names corresponding to ...
# (a) Nucleotide BDM
NBDM_feats = ['NucleotideBDM_' + str(i) for i in range(1, 1 + TAR_SEQ_LEN)]
# (b) GC147
GC_feats = ['GCContent_' + str(i) for i in range(1, 1 + TAR_SEQ_LEN)]
# (c) NuPoP (Affinity)
NuPoP_Aff_feats = ['NuPoP_Affinity_147_h'] + \
            ['NuPoP_Affinity_147_h.' + str(i) for i in range(1, TAR_SEQ_LEN)]
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

# We select NUM_SAMPLES random data points...
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

    # Make predictions for these artificial datapoints
    preds = xgb_model.predict(vary_df)

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
#   - low Nucleotide BDM value corresponds to low predicted CRISPR-Cas9 cleavage
#     activity value
#   - increasing GC147 increases predicted CRISPR-Cas9 cleavage activity value
#   - high NuPoP (Affinity) value corresponds to low predicted CRISPR-Cas9
#     cleavage activity value
