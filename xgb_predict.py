X_df = pd.read_csv('/content/drive/MyDrive/nuc_occup/sample_script/data/explainset_xgboost_noseq_engepi_nuc_MNase.csv')
new_xgb_loc = '/content/drive/MyDrive/nuc_occup/sample_script/models/xgb_noseq_engnucepi_model_post_xgb1.6.json'
new_xgb_model = xgb.XGBRegressor()
new_xgb_model.load_model(new_xgb_loc)
preds = new_xgb_model.predict(X_df)
out_loc = '/content/drive/MyDrive/nuc_occup/sample_script/data/xgb_preds.csv'
pd.DataFrame(preds).to_csv(out_loc, index=None)
print('Predicted CRISPR-Cas9 Activities: ' + str(preds))
