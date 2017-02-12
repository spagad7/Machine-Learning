Cross-Validation
================
Note: I am using sklearn and numpy libraries

my_cross_val.py
===============
This is a code for performing k-fold cross validation on a given dataset. 
For mode info on cross-validation check: https://en.wikipedia.org/wiki/Cross-validation_(statistics)


my_train_test.py
================
This is a code to split given dataset into train and test set and fit a model on train set and predict on test set.


generateDatasets.py
===================
This is a code to generate Boston50, Boston75 and Digits dataset. 

Boston50: This is a clone of Boston dataset in sklearn, but the target values are set to 1 if the target value is greater than 50th percentile, and 0 if the target value is below 50th percentile. 

Boston75: This is a clone of Boston dataset in sklearn, but the target values are set to 1 if the target value is greater than 75th percentile, and 0 if the target value is below 75th percentile. 

Digits: This is a clone of digits dataset in sklearn, with no modifications.


run_my_cross_val.py
===================
This is a wrapper code for my_cross_val. It can be run both with and without arguments.

Without Arguments: It will run my_cross_val with LinearSVC, SVC and Logistic Regression on Boston50, Boston75 and Digits datasets

With Arguments: Pass 3 arguments: 
Arg1: <String>  modelname       [LinearSVC, SVC, LogisticRegression] 
Arg2: <String>  dataset-name    [Boston50 | Boston75 | Digits] 
Arg3: <int>     k               k-fold value



run_my_train_test.py
=======
This is a wrapper code code for wrapper code for my_train_test. It can be run both with and without arguments.

Without Arguments: It will run my_train_test with LinearSVC, SVC and Logistic Regression on Boston50, Boston75 and Digits datasets

With Arguments: Pass 4 arguments: 
Arg1: <String>  modelname       [LinearSVC, SVC, LogisticRegression] 
Arg2: <String>  dataset-name    [Boston50 | Boston75 | Digits] 
Arg3: <float>   pi              Value between 0 and 1 speicifying what fraction of dataset to consider for training
Arg4: <int>     k               Number of times to train and test the model



run_rand_quad_proj.py
=====================
This is a wrapper code code for running rand_proj and quad_proj. After calling rand_proj and quad_proj it runs my_cross_val to perform cross validation on model.It can be run both with and without arguments.

Without Arguments: it will run rand_proj and quad_proj with LinearSVC, SVC and LogisticRegression on digits dataset

With Arguments: Pass 4 arguments: 
Arg1: <String>  modelname        [LinearSVC, SVC, LogisticRegression] 
Arg2: <String>  dataset-name     [Boston50 | Boston75 | Digits] 
Arg3: <String>  projection       [rand_proj | quad_proj]
Arg4: <int>     k                k-fold value
