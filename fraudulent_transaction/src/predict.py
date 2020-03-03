import sys
import numpy as np
import pandas as pd
from joblib import dump, load
from DataLoader import DataLoader

data_path = sys.argv[1]
model_name = sys.argv[2]

df = pd.read_csv(data_path) 
scaler = load('../data/output/models/'+model_name+'_scaler.joblib')
model = load('../data/output/models/'+model_name+'.joblib')

scaled_df = scaler.transform(df)
prediction = model.predict(scaled_df)
pd.DataFrame(prediction).to_csv('../data/output/predictions/'+model_name+'_prediction.csv')
print("Done ! The prediction is saved at "+'../data/output/predictions/'+model_name+'_prediction.csv')
