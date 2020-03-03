import numpy as np
import pandas as pd
import random

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

import seaborn as sns

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
    
def save_corr(df):
    ax = sns.heatmap(df.corr(), vmin = -1, vmax = 1, cmap='coolwarm')
    ax.figure.savefig('../data/output/corr.jpg')

def plot_9_violinplot(df):
    v_nums = [i for i in range(1, 29)]
    v_choosed = random.choices(v_nums, k=9)
    idx = 0

    fig,ax =  plt.subplots(3,3, figsize=(10,10))
    for i in range(3):
        for j in range(3):
            sns.violinplot(df["V"+str(v_choosed[idx])], ax = ax[i][j])
            idx += 1
    fig.savefig('../data/output/9_violinplot.jpg')