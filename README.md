# crispr-cas9-epigenetics

This Github repository contains sample Python scripts and CSV datasets for running the XGBoost and CNN models produced in the paper "Comprehensive computational analysis of epigenetic descriptors affecting CRISPR-Cas9 off-target activity".

## Models
This repository contains the following models in the "model" folder:
- XGBoost model: xgb_noseq_engnucepi_model_post_xgb1.6.json
- CNN model: torch_noseq_engnucepi_model.pt

## Data
This repository contains the following CSV datasets in the data folder:

## Scripts
This repository contains the following Python scripts
* xgb_predict_one.py: takes a CSV file "example_input.csv" containing a single point as input, and produces a CRISPR-Cas9 off-target activity prediction based on the  XGBoost model
* xgb_predict.py: takes a CSV file ""

The modelling code in cnn.py is adopted from https://github.com/florianst/picrispr.

# Contacts
jeffrey.kelvin.mak@cs.ox.ac.uk or peter.minary@cs.ox.ac.uk
