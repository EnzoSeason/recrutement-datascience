import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

def delete_outliers(Xtrain, ytrain, variable):
    '''
    find IQR, and delete outliers
    
    parameters:
    -- df : pandas Dataframe which contains all the data
    -- variable : the variable we want to treat
    -- inplace : If True, do operation inplace and return None.
    '''
    
    df = pd.concat([Xtrain,ytrain], axis=1)
    v_df = df[variable]
    q25, q75 = np.percentile(v_df, 25), np.percentile(v_df, 75)
    iqr = q75 - q25
    # print('{}, Quartile 25: {} | Quartile 75: {} | IQR: {}'.format(variable, q25, q75, iqr))

    lower, upper = q25 - iqr * 1.5, q75 + iqr * 1.5
    
    new_df = df.drop(df[(df[variable] > upper) | (df[variable] < lower)].index)
    y = new_df['Class']
    X = new_df.drop(columns=['Class'])
    
    return X, y

def CV_SMOTE(original_Xtrain, original_ytrain, model, params, n_iter, K):
    '''
    Cross-Validation with SMOTE, RandomizedSearchCV
    
    Parameters:
    -- model: ML model
    -- params: hyper parameters to be fine-tuned
    -- n_iter: numbers of iteration in RandomizedSearchCV
    -- K: number of folds in CV
    '''
    
    # use RandomizedSearchCV for fine-tuning hyper parametres
    cv = RandomizedSearchCV(model, params, n_iter=4)
    # make sure that each fold has the data from all the classes
    sss = StratifiedKFold(n_splits=K, random_state=None, shuffle=False)
    
    for CV_train, CV_test in sss.split(original_Xtrain, original_ytrain):
        smt = SMOTE()
        # oversampling
        oversampling_Xtrain, oversampling_ytrain = smt.fit_sample(original_Xtrain.iloc[CV_train], original_ytrain.iloc[CV_train])
        # fine-tuning
        search = cv.fit(oversampling_Xtrain, oversampling_ytrain)
        best_est = search.best_estimator_
        # prediction
        prediction = best_est.predict(original_Xtrain.values[CV_test])

        print(search.best_params_)
        print(classification_report(original_ytrain.values[CV_test], prediction))