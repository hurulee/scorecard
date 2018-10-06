# Credit scoring projects
### note: reject inference is not included yet

This repo includes the following credit scorecard model:

1. *scorecard_gr_log_woe.ipynb*

   * The data used in this notebook is downloaded from https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
   
   * Binning is performed automatically such the that relationship between bins of numerical variables is monotonic in the weight of evidence (see https://medium.com/@sundarstyles89/weight-of-evidence-and-information-value-using-python-6f05072e83eb)
   
   * This scorecard model uses logistic regression, where the original training data has been converted to binned weight of evidence values. 
   
   * Attribute scores are a function of the estimated coefficients and weight of evidence.
   
   * Characteristics for adverse action codes are selected based on attribute score relative to neutral score. 
