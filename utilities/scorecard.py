# -*- coding: utf-8 -*-

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

def scorecard(df, model, woe_iv_table, points0, odds0, pdo):
    """
    Calculates the total score of each borrower observation in the data frame.
    Score calculation based on chapter four of "Intelligent Credit Scoring: 
    Building and Implementing Better Credit Risk Scorecards" (2nd edition, 2017) by 
    Naeem Siddiqi
    
    Parameters
    ----------
    
    df : pandas data frame
        WOE-converted binned training dataframe - woe_conversion function output
    
    model : sklearn object
        fitted model object
    
    woe_iv_table : pandas data frame
        Weight of evidence / information value table and other data used to 
        calculate WOE and IV for variables in dataset - woe_analysis function
        output
    
    points0 : int
        base score used to calculate offset
        
    odd0 : int
        base odds used to calculate offset
        
    pdo : int
        "points to double odds" - scaling factor used in credit scoring models 
        built with logistic regression
    
    Return
    ------
    
    score_tab : pandas data frame
        score card summary table 
        
    df_copy : pandas data frame
        training dataframe with the original WOE input values for every characteristic 
        in the model, converted into the corresponding scores using the formulae 
        provided in Siddiqi (2017). The resulting score for every borrower's attribute 
        for every characteristic is summed across the row to generate the total score. 
    
    """
    points0=points0
    odds0=odds0 
    pdo=pdo
    
    #scored dataframe
    df_copy = df.copy(deep=True)
    n = len(df_copy.columns)
    # alpha = logreg_clf.intercept_
    if pdo > 0:
        factor = pdo/np.log(2)
    else:
        factor = -pdo/np.log(2)

    offset = points0 - factor*np.log(odds0) #log(odds0/(1+odds0))

    coef_series = pd.Series(model.coef_[0], index=np.array(df_copy.columns)).loc[lambda x: x != 0]

    col_names = list(df_copy.columns)
    col_names_short = list([re.sub('_woe$', '', i) for i in df_copy.columns])

    sum_str_list = []

    for i in range(0, len(col_names)):
        coef = coef_series[i]
        col_name_score = (col_names_short[i] + '_score')
        
        df_copy[col_name_score] = -(df_copy[col_names[i]] * coef + (model.intercept_  / n )) * factor + (offset / n)
        df_copy = df_copy.drop([col_names[i]], axis=1)

        char_str = "'%s'" % col_name_score
        com_str = "df_copy[" + char_str + "]" 
        sum_str_list.append(com_str)

    full_conds = '+'.join(sum_str_list)

    df_copy['neutral'] = np.asscalar(-((model.intercept_ / n) * factor) + (offset / n))

    df_copy['total_score'] = eval(full_conds) 
    
    coef_df = coef_series.to_frame()
    coef_df.reset_index(level=0, inplace=True)
    coef_df['index'] = coef_df['index'].apply(lambda x: x.split('_woe')[0])
    coef_df.rename(columns={'index': 'variable', 0:'coef'}, inplace=True)

    woe_iv_table_copy = woe_iv_table.copy(deep=True)
    woe_iv_table_copy = woe_iv_table_copy[['VAR_NAME', 'bucket', 'WOE']]

    score_tab = pd.merge(woe_iv_table_copy, coef_df, how ='inner', left_on =['VAR_NAME'],  right_on = ['variable'])
    score_tab['score'] = -(score_tab['WOE'] * score_tab['coef'] + (model.intercept_ / n)) * factor + (offset / n)
    score_tab = score_tab.drop(['VAR_NAME'], axis=1)
    score_tab = score_tab[['variable', 'bucket', 'WOE', 'score']]

    return score_tab, df_copy
