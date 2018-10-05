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


# see https://medium.com/@sundarstyles89/weight-of-evidence-and-information-value-using-python-6f05072e83eb
# and https://github.com/Sundar0989/WOE-and-IV/blob/master/WOE_IV.ipynb for WOE and IV calculation
# he got the order of dist_good and dist_bad reversed in WOE and IV


def woe_conversion(df, woe):
    """
    Converts the values of each variable for each borrower from its original
    value into the weight of evidence (WOE) values of the variable bin that 
    the input value is in.
    
    Parameters
    ----------
    
    df : pandas dataframe
        Cleaned explanatory variable training / testing / validation data frame
        that will be used to fit the model.
        
    woe : pandas dataframe
        WOE / IV table that is output from woe_analysis() function
        
    Return
    ------
    
    df_copy : pandas dataframe
        Converted dataframe, from original input values to corresponding WOE values
        
    """
    df_copy = df.copy()
    woe_df = woe.copy()

    var_list = list(df_copy)

    for i in range(0,len(var_list)):
        var_str = "'%s'" % var_list[i]
        var_woe = woe_df.loc[woe_df['VAR_NAME'] == var_list[i]].copy()

        var_woe['max_range'] = var_woe['MAX_VALUE']

        if np.issubdtype(df_copy[var_list[i]], np.number) and\
        (len(Series.unique(df_copy[var_list[i]])) > 2):
            var_woe['min_range'] = var_woe.groupby('VAR_NAME')['MAX_VALUE'].shift(1)
            var_woe.loc[var_woe['MIN_VALUE'].isnull(), 'min_range'].isnull()
            var_woe.loc[var_woe['min_range'].isnull(), 'min_range'] = var_woe['MIN_VALUE']
        else:
            var_woe['min_range'] = var_woe['MIN_VALUE']
 
        var_woe_clean = var_woe[var_woe['MIN_VALUE'].notnull()]
        var_woe_null = var_woe[var_woe['MIN_VALUE'].isnull()]

        if not var_woe_null.empty:
            woe_null = var_woe_null.iloc[0]['WOE']
        else:
            woe_null = np.nan

        min_value_list = var_woe_clean['min_range'].tolist()
        max_value_list = var_woe_clean['max_range'].tolist()
        choices = var_woe_clean['WOE'].tolist()

        cond_str_list = []

        N = len(min_value_list)
        
        for j in range(0,len(min_value_list)):  
            
                #condition for binary indicator variables
            if np.issubdtype(df_copy[var_list[i]], np.number) and \
            (len(Series.unique(df_copy[var_list[i]])) == 2) and \
            min_value_list[j] == max_value_list[j]: 
                com_str = "(df_copy[" + var_str + "] ==" + str(min_value_list[j]) + ")"
                
            elif np.issubdtype(df_copy[var_list[i]], np.number):
                if j == 0:
                    com_str = "(df_copy[" + var_str + "] <=" + str(max_value_list[j]) + ")"
                elif j == (N-1):
                    com_str = "(df_copy[" + var_str + "] >" + str(min_value_list[j]) + ")"
                else:
                    com_str = "(df_copy[" + var_str + "] >" + str(min_value_list[j]) + ") & (df_copy[" + var_str + "] <=" + str(max_value_list[j]) + ")"

            else:
                char_str = "'%s'" % min_value_list[j]
                com_str = "(df_copy[" + var_str + "] ==" + char_str + ")"
                
            cond_str_list.append(com_str)

        full_conds = ','.join(cond_str_list)

        conditions = eval(full_conds)
        var_woe_label = var_list[i] +'_woe'
        
        df_copy[var_woe_label] = np.select(conditions, choices, default =woe_null)
        df_copy = df_copy.drop([var_list[i]], axis=1)
    return df_copy


def woe_graph(df, is_numeric):
    """
    This function is called within the woe_analysis() function.
    
    Plots the optimal binning of variables in training / testing / validation dataframe
    Figure and table with buckets, min observed value and max observed value for each 
    variable is generated. 
    
    Parameters
    ----------
    df : pandas dataframe
        output data frame from mono_bin or char_bin functions. It is a dataframe for 
        one optimally binned variable with information to generate corresponding graph
        
    is_numeric : boolean
       True = if variable type is int or float and not binary indicator variable
       (i.e., only two values)
       False = variable type is obj or binary indicator
       
    Return
    ------
    N/A
    
    """
    binx = df.copy()
    # binx = binx.loc[binx['VAR_NAME'] == 'property_type']
    total = binx['COUNT'].sum()
    binx['event_dist'] = binx['EVENT'] / total
    binx['non_event_dist'] = binx['NONEVENT'] / total
    binx['dist'] = binx['COUNT'] / total

    if is_numeric == True:
        binx['MIN_VALUE_STR'] = binx['MIN_VALUE'].astype(str)
        binx['MAX_VALUE_STR'] = binx['MAX_VALUE'].astype(str)
        binx['bin'] = binx[['MIN_VALUE_STR', 'MAX_VALUE_STR']].apply(lambda x: ','.join(x), axis=1)
    else:
        binx['bin'] = binx['MIN_VALUE']

    binx = binx.reset_index(drop=True)

    ## y_right_max and y_right_min
    y_right_max = np.ceil(binx['WOE'].max()*10)
    if y_right_max % 2 == 1: 
        y_right_max=y_right_max+1

    if y_right_max - binx['WOE'].max()*10 <= 0.3: 
        y_right_max = y_right_max+2

    y_right_max = y_right_max/10

    y_right_min = np.ceil(binx['WOE'].min()*10)
    if y_right_min % 2 == 1: 
        y_right_min=y_right_min-1

    if abs(y_right_min - binx['WOE'].min()*10) <= 0.3: 
        y_right_min = y_right_min-1

    y_right_min = y_right_min/10

    ## y_left_max
    y_left_max = np.ceil(binx['dist'].max()*10)/10
    if y_left_max>1 or y_left_max<=0 or y_left_max is np.nan or y_left_max is None: 
        y_left_max=1

    width = 0.35       # the width of the bars: can also be len(x) sequence
    ind = np.arange(len(binx.index))    # the x locations for the groups
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    p1 = ax1.bar(ind, binx['non_event_dist'], width, color=(24/254, 192/254, 196/254))
    p2 = ax1.bar(ind, binx['event_dist'], width, bottom=binx['non_event_dist'], color=(246/254, 115/254, 109/254))

    for i in ind:
        ax1.text(i, binx.loc[i,'dist']*1.06, str(round(binx.loc[i,'dist']*100,1))+'%, '+str(binx.loc[i,'COUNT']), ha='center')
    
    ax2.plot(ind, binx['WOE'], marker='o', color='blue')
        
    # settings
    ax1.set_ylabel('Bin count distribution')
    ax2.set_ylabel('Weight of evidence', color='blue')
    ax1.set_yticks(np.arange(0, y_left_max+0.2, 0.1))
    ax2.tick_params(axis='y', colors='blue')

    plt.xticks(ind, binx['bin'], rotation = 75)
    title_string = binx.loc[0,'VAR_NAME'] + "  (iv: " + str(round(binx.loc[0,'IV'],4)) + ")" 
    plt.title(title_string, loc='center')
    plt.legend((p2[0], p1[0]), ('bad', 'good'), loc='best')
    # show plot 
    plt.show()


def woe_analysis(df1, target, max_bin, force_bin):
    """
    wrapper function for mono_bin, char_bin, and woe_graph functions.
    This will automatically construct bins for each variable. For numerical
    variables, it will create bins such that the WOE relationship between bins 
    is monotonic.
    
    Parameters
    ----------
    
    df1 : pandas dataframe
        training dataset
        
    target : pandas series
        target vector
        
    max_bin : int
        the maximum number of bins (categories) for numeric variable binning. 
        
    force_bin : int
        For some numeric variables, the mono_bin function may produce only one 
        category while binning. ‘force_bin’ ensures that at least produces two
        categories will be produced.
        
    Return
    ------
    
    iv_df : pandas dataframe
        Weight of evidence / information value table and other data used to 
        calculate WOE and IV for variables in dataset
    
    iv : pandas dataframe
        Information value table for variables in dataset
    
    """
    max_bin = max_bin
    force_bin = force_bin
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i], max_bin, force_bin)
                conv["VAR_NAME"] = i
                count = count + 1
                woe_graph(conv, True)
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                conv = conv.sort_values(by='WOE', ascending=False)
                woe_graph(conv, False)
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return iv_df, iv



def mono_bin(Y, X, max_bin, force_bin):
    """
    binning function for int and float type variables, and not binary indicator variable
    
    Parameters
    ----------
    
    Y : pandas series
        target vector
        
    X : pandas dataframe
        training dataset
        
    max_bin : int
        the maximum number of bins (categories) for numeric variable binning. 
        
    force_bin : int
        For some numeric variables, the mono_bin function may produce only one 
        category while binning. ‘force_bin’ ensures that at least produces two
        categories will be produced. 
        
    Return
    ------
    
    d3 : pandas dataframe
        Weight of evidence / information value table and other data used to 
        calculate WOE and IV for variable i in dataset
       
    """
    n = max_bin
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": \
                           pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
        
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    print(d3)
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    d3 = d3.drop(d3[d3.COUNT == 0].index)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_NON_EVENT/d3.DIST_EVENT)
    d3["IV"] = (d3.DIST_NON_EVENT-d3.DIST_EVENT)*np.log(d3.DIST_NON_EVENT/d3.DIST_EVENT)
    d3["VAR_NAME"] = "VAR"

    d3['max_range'] = d3['MAX_VALUE']
    d3["min_range"] = d3.groupby('VAR_NAME')['MAX_VALUE'].shift(1)
    d3.loc[d3['min_range'].isnull(), 'min_range'] = -np.inf
    d3.loc[d3['MIN_VALUE'].isnull(), 'min_range'] = np.nan
    
    _max = d3.loc[d3['max_range'] != np.nan, 'max_range'].max()
    d3.loc[d3['max_range'] == _max, 'max_range'] = np.inf  

    d3['bucket'] = '(' + d3['min_range'].astype(str) + ', ' + d3['max_range'].astype(str) + ']'
    d3.loc[d3['bucket'] == '(nan, nan]', 'bucket'] = 'missing' 
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'min_range', 'max_range', 'COUNT', 'EVENT', 'EVENT_RATE', \
             'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT', 'bucket', 'WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
    """
    binning function for obj type variables, and binary indicator variable
    
    Parameters
    ----------
    
    Y : pandas series
        target vector
        
    X : pandas dataframe
        training dataset
        
    Return
    ------
    
    d3 : pandas dataframe
        Weight of evidence / information value table and other data used to 
        calculate WOE and IV for variable i in dataset
       
    """    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    print(d3)
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    d3 = d3.drop(d3[d3.COUNT == 0].index)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_NON_EVENT/d3.DIST_EVENT)
    d3["IV"] = (d3.DIST_NON_EVENT-d3.DIST_EVENT)*np.log(d3.DIST_NON_EVENT/d3.DIST_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3['min_range'] = d3['MIN_VALUE']
    d3['max_range'] = d3['MAX_VALUE']
    d3['bucket'] = d3['MIN_VALUE']
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'min_range', 'max_range', 'COUNT', 'EVENT', 'EVENT_RATE', \
             'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT', 'bucket', 'WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

