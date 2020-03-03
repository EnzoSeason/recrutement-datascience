# load data
import numpy as np 
import pandas as pd
# scale data
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataLoader:
    def __init__(self, data_path, sampling_type='SMOTE', scaler='RobustScaler'):

        # load data
        self.raw_data = pd.read_csv(data_path)

        # split into train/test set
        X = self.raw_data.drop('Class', axis=1)
        y = self.raw_data['Class']
        self.raw_train_X, self.raw_test_X, self.raw_train_y, self.raw_test_y = train_test_split(X, y, test_size=0.2, random_state=0)

        # sampling
        if sampling_type == 'SMOTE':
            self.sampling_type = 'SMOTE'
            smt = SMOTE(random_state=0)
            self.sampled_train_X, self.sampled_train_y = smt.fit_sample(self.raw_train_X, self.raw_train_y)
        if sampling_type == 'None':
            self.sampled_train_X = None

        # scale data
        if isinstance(self.sampled_train_X, pd.DataFrame):
            train_X = self.sampled_train_X
        else:
            train_X = self.raw_train_X
            
        if scaler == 'RobustScaler':
            self.scaler_type = 'RobustScaler'
            self.scaler = RobustScaler().fit(train_X)
            self.scaled_train_X = self.transform(train_X)
            self.scaled_test_X = self.transform(self.raw_test_X)

    
    def transform(self, df):
        '''
        Scale data with DataLoader's scaler
        '''
        if self.scaler:
            scaled_data = self.scaler.transform(df)
            return pd.DataFrame(scaled_data, columns=df.columns)
        else:
            return df

        


