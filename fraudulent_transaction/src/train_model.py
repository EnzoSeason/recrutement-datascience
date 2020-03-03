import sys
# build model
from sklearn.ensemble import RandomForestClassifier
# evaluate model
from sklearn.metrics import classification_report
# save model
from joblib import dump, load
# personal tools
from DataLoader import DataLoader

data_path = sys.argv[1]
model_name = sys.argv[2]

# load data
data_loader = DataLoader(data_path)
print(data_path + 'is loaded !')

# train the model
print('Training Model ...')
model =  RandomForestClassifier()
model.fit(data_loader.scaled_train_X, data_loader.sampled_train_y)
print('Done !')

# evalue the model
prediction = model.predict(data_loader.scaled_test_X)
print(classification_report(data_loader.raw_test_y, prediction))

# save the model
dump(data_loader.scaler, '../data/output/models/'+model_name+'_scaler.joblib')
dump(model, '../data/output/models/'+model_name+'.joblib')
print('Model is save at'+'/data/output/models/'+model_name+'_scaler.joblib')