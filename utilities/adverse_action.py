# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
from collections import Counter
from copy import deepcopy

import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

 
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, \
    AdaBoostClassifier, GradientBoostingRegressor, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.model_selection import train_test_split #to create validation data set
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from pandas.tools.plotting import scatter_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, Imputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error, make_scorer
from sklearn import preprocessing

import statsmodels.formula.api as sm

from sklearn import model_selection

def adverse_action(scored_data):
    """
    Generates the top four adverse action codes for each borrower observation in the 
    input dataframe. For each borrower observation, this function will generate four 
    columns, where the first column is the characteristic with smallest
    difference from the neutral score, the second column is the characteristic with the
    second smallest difference from the neutral score, etc. The last (fifth) column 
    is the borrowers credit score from the scored_data dataframe. 
    
    Parameters
    ----------
    
    scored_data : pandas dataframe
        The data frame with borrower score column that is outputted from scorecard() function
        
    Return
    ------
    
    reason_tab : pandas dataframe
        Dataframe with five columns, indexed by borrower observation number. 
        The first column lists the characteristic with the least (presumably negative)
        difference from the neutral score, the second column list the characteristic
        with the second least difference from the neutral score, etc. The last column
        lists the borrower's total score (from the scored_data dataframe). 
    
    """
    
    df_copy = scored_data.copy(deep=True)
    col_names = list(df_copy.columns)

    col_names = col_names[:len(col_names)-2]
    col_names_short = list([re.sub('_score$', '', i) for i in col_names])

    diff_var_list = []

    for i in range(0, len(col_names)):
        df_copy[col_names_short[i]] = df_copy[col_names[i]] - df_copy['neutral']
        diff_var_list.append(col_names_short[i])

    score = df_copy['total_score']
    df_copy = df_copy[diff_var_list]

    arr = np.argsort(df_copy.values, axis=1)
    reason = pd.DataFrame(df_copy.columns[arr], index=df_copy.index)

    reason1_4 = reason[[0, 1, 2, 3]]
    reason1_4 = reason1_4.rename(index=str, columns={0: 'reason_1', 1: 'reason_2', 2: 'reason_3', 3: 'reason_4'})
    reason_tab = pd.concat([reason1_4, score], axis=1)
    
    return reason_tab