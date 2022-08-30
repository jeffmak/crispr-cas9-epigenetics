import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# Nucleotide BDM, GC147 and NuPoP (Affinity) are identified as computed nucleosome organization-related scores. Thus, in this Python script, we perturb the values of these scores, and see how such perturbations affect XGBoost's predictions.

# This Python script uses the following data and model files:
# 1. An input CSV file containing 2000 random experimental/augmented datapoints from the crisprSQL dataset in the paper
csv_loc = 'data/explainset_xgboost_noseq_engepi_nuc_MNase.csv'
# 2. The XGBoost tree model used for making CRISPR-Cas9 activity prediction
xgb_loc = 'models/xgb_noseq_engnucepi_model_post_xgb1.6.json'

# The plots generated will be saved in these locations
out_dict = {'Nucleotide BDM': 'out/xgb_perturb_nucleotide_bdm.pdf',
            'GC147': 'out/xgb_perturb_gc147.pdf',
            'NuPoP (Affinity)': 'out/xgb_perturb_nupop_affinity.pdf'}


# Read the input CSV file
X_df = pd.read_csv(csv_loc)

# Create and load the XGBoost model
# Use XGBRegressor since CRISPR-Cas9 activity values are continuous
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(xgb_loc)

# These are the column/feature names corresponding to ...
# (a) Nucleotide BDM
NBDM_feats = ['NucleotideBDM_' + str(i) for i in range(1, 24)]
# (b) GC147
GC_feats = ['GCContent_' + str(i) for i in range(1, 24)]
# (c) NuPoP (Affinity)
NuPoP_Aff_feats = ['NuPoP_Affinity_147_h'] + \
                  ['NuPoP_Affinity_147_h.' + str(i) for i in  range(1,23)]
# Put these features into a dictionary
feats_dict = {'Nucleotide BDM': NBDM_feats,
              'GC147': GC_feats,
              'NuPoP (Affinity)': NuPoP_Aff_feats}

# Set the perturbation ranges for each computed score
# to be within the minimum and maximum value of the respective features,
# i.e., we perturb within the ranges of the dataset.
ranges_dict = {'Nucleotide BDM': (X_df[NBDM_feats].min(axis=1).min(),
                                  X_df[NBDM_feats].max(axis=1).max()),
              'GC147': (X_df[GC_feats].min(axis=1).min(),
                        X_df[GC_feats].max(axis=1).max()),
              'NuPoP (Affinity)': (X_df[NuPoP_Aff_feats].min(axis=1).min(),
                                   X_df[NuPoP_Aff_feats].max(axis=1).max())}

# Generate plots for each of the three computed scores
for feat_name, feats in feats_dict.items():
  min_val, max_val = ranges_dict[feat_name]
  print("{}'s' value ranges from {:.2f} to {:.2f}".format(feat_name, min_val, max_val))

  # We select five random data points...
  for _ in range(5):
    point_chosen = X_df.sample().iloc[0]

    # Create a Pandas dataframe for the random datapoint
    example_df = pd.DataFrame(point_chosen).transpose()

    # Determine the values to perturb to
    varied_vals = np.linspace(min_val, max_val, 20)

    # Create the artificial datapoints containing the perturbed values
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
  plt.xlabel("Perturbed {} Value".format(feat_name))
  # the y-axis's name
  plt.ylabel("Predicted CRISPR-Cas9 Activity")

  # Save the plot
  plt.savefig(out_dict[feat_name])

  # close the current figure
  plt.close()
 
# The script prints:
# Nucleotide BDM's' value ranges from 183.95 to 481.80
# GC147's' value ranges from 0.20 to 0.86
# NuPoP (Affinity)'s' value ranges from 0.51 to 1.96
