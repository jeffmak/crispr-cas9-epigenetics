# Code for the LeNup (H3Q85C) nucleosome occupancy prediction model

This folder contains sample Python scripts for running the LeNup (H3Q85C) model, which predicts nucleosome occupancy. The model is based on the CNN described in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5946947/, but is retrained on H3Q85C chemical cleavage yeast nucleosome positioning data, specifically GSM2561057 from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5807854/. The model is implemented using PyTorch.


## Model
This repository contains the following models in the ```model``` folder:
|      File       |  Description   |
| ----------------|--------------- |
| lenup_h3q85c.th | CNN model.     |

## Data
This repository contains the following CSV dataset in the ```data``` folder:
| File | Description |
| --------------|------------ |
| example_offtargets.csv | Contains the first five CRISPR-Cas9 off-target sites from the crisprSQL database. Each row represents a single datapoint, and includes the associated 169bp context sequence. |

## Sample Script
The script ```lenup_pred.py``` must be run from this subdirectory in order for it to work.

| File | Runnable | Description |
| -------| -------- | -------------|
| lenup_nn.py | no | Contains PyTorch for LeNup's model architecture. |
| lenup_pred.py | yes | Makes base pair-resolved nucleosome occupancy predictions for the five CRISPR-Cas9 off-target activity datapoints in ```data/example_offtargets.csv```, and saves them to ```data/output.csv```. |
| nn_pred.py | no | Wrapper class for the LeNup model. Handles batching and one-hot encoding of input sequences. |
