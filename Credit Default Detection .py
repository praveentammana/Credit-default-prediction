#!/usr/bin/env python
# coding: utf-8

# # Import Libs

# In[11]:


# Libraries
# Data Import
import urllib.request

# Data preparetion
import pandas as pd
import numpy as np

# Graphs
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

# Data Preprocession
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Data Normalization

# Metrics
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score # Cross Validation
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

# Hyperparamter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score 


# Model Tuning

# Models
from sklearn.dummy import DummyClassifier # Baseline Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from keras import models
from keras import layers


# # Import Data

# In[12]:


credit_data = pd.read_csv("C:/Users/praveen/Downloads/UCI_Credit_Card.csv")
credit_data = credit_data.iloc[1:,:].drop(columns = "ID").reset_index().drop(columns = "index")
credit_data.head(20)


# In[13]:


credit_data.columns


# In[14]:


print(type(credit_data))
print (credit_data.dtypes)
credit_data = credit_data.astype(int)
print (credit_data.dtypes)


# # EDA

# In[15]:


# Check for whether there is missing values
#print(credit_data.isnull().sum())
print(credit_data.info())
# Description of the data
credit_data.describe()


# In[19]:


num_data = len(credit_data["default.payment.next.month"])
num_def = len(credit_data[credit_data["default.payment.next.month"]== 1])
percent_def = len(credit_data[credit_data["default.payment.next.month"]== 1])/len(credit_data["default.payment.next.month"])
percent_non_def = 1- percent_def
label = ["Default", "Non-Default"]
percent = [percent_def, percent_non_def]
plt.bar(label, percent)
plt.ylabel('Percentage %')
plt.show()


# In[21]:


credit_data["sum_pay"] = credit_data.PAY_AMT1+credit_data.PAY_AMT2+credit_data.PAY_AMT3+credit_data.PAY_AMT4+credit_data.PAY_AMT5+credit_data.PAY_AMT6
credit_data["sum_pay"] = pd.to_numeric(credit_data["sum_pay"])

fig, ax = plt.subplots(1,2,figsize =(12,4), sharey= True,sharex=True)
from scipy.stats import norm
default_set = credit_data[credit_data["default.payment.next.month"]== 1]
non_default_set = credit_data[credit_data["default.payment.next.month"]== 0]
sns.distplot(default_set.LIMIT_BAL/1000, fit = norm, ax = ax[0], color="r")
ax[0].set(title= "Default creditor's limited credits")
ax[0].set_xlabel("The limited Credits (in thousands)")


sns.distplot(non_default_set.LIMIT_BAL/1000, fit = norm, ax = ax[1],label ="Non-default creditor's limited credits" )
ax[1].set(title= "Non-default creditor's limited credits")
ax[1].set_xlabel("The limited Credits (in thousands)")
plt.show()


# In[22]:


#d_sum_pay = dedefault_set["sum_pay"])
plt.figure(figsize= (15,10))
sns.distplot(default_set["sum_pay"]/100,bins = 200)
sns.distplot(non_default_set["sum_pay"]/100,bins = 200)
plt.legend(loc = "best")
plt.show()


# In[24]:


credit_data["Defualt_status"] = credit_data["default.payment.next.month"]
credit_data["Defualt_status"].replace(1,"Defualt", inplace = True)
credit_data["Defualt_status"].replace(0,"Non-defualt", inplace = True)

with sns.axes_style('white'):
    g = sns.jointplot("AGE", "LIMIT_BAL", data=credit_data ,kind='hex')
    p = sns.jointplot("AGE", "LIMIT_BAL", data=credit_data ,hue = "Defualt_status")
    plt.show()


# In[25]:


print(credit_data.corr())


# In[26]:


plt.subplots(figsize=(25,17))
sns.heatmap(credit_data.corr(), annot=True)
plt.show()


# In[27]:


sns.set(rc={'figure.figsize':(15.7,8.27)})
#plt.figure(figsize= (40,15))
sns.pairplot(credit_data[["LIMIT_BAL",	"SEX" ,	"EDUCATION",	"MARRIAGE",	"AGE","Defualt_status"]], hue = "Defualt_status")
plt.show()


# # Data Preprocessing
# 

# In[28]:


credit_data["pay_diff1"]  = credit_data["PAY_0"].sub(credit_data["PAY_2"], axis = 0)
credit_data["pay_diff2"]  = credit_data["PAY_2"].sub(credit_data["PAY_3"], axis = 0)
credit_data["pay_diff3"]  = credit_data["PAY_3"].sub(credit_data["PAY_4"], axis = 0)
credit_data["pay_diff4"]  = credit_data["PAY_4"].sub(credit_data["PAY_5"], axis = 0)
credit_data["pay_diff5"]  = credit_data["PAY_5"].sub(credit_data["PAY_6"], axis = 0)

credit_data["AMT_diff1"]  = credit_data['BILL_AMT1'].sub(credit_data['BILL_AMT2'], axis = 0)
credit_data["AMT_diff2"]  = credit_data['BILL_AMT2'].sub(credit_data['BILL_AMT3'], axis = 0)
credit_data["AMT_diff3"]  = credit_data['BILL_AMT3'].sub(credit_data['BILL_AMT4'], axis = 0)
credit_data["AMT_diff4"]  = credit_data['BILL_AMT4'].sub(credit_data['BILL_AMT5'], axis = 0)
credit_data["AMT_diff5"]  = credit_data['BILL_AMT5'].sub(credit_data['BILL_AMT6'], axis = 0)

#credit_data["sum_pay0_1"] = credit_data.PAY_0+credit_data.PAY_2+credit_data.PAY_3+credit_data.PAY_4+credit_data.PAY_5+credit_data.PAY_6


# In[30]:


# Convert the DataFrame object into NumPy array otherwise you will not be able to impute
#cols = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",	"PAY_0",	"PAY_2",	"PAY_3",	"PAY_4",	"PAY_5",	"PAY_6",	"BILL_AMT1",	"BILL_AMT2",	"BILL_AMT3",	"BILL_AMT4",	"BILL_AMT5",	"BILL_AMT6",	"PAY_AMT1",	"PAY_AMT2",	"PAY_AMT3",	"PAY_AMT4",	"PAY_AMT5",	"PAY_AMT6",	"default payment next month"]
#cols = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",		"PAY_0",	"PAY_2",	"PAY_3",	"PAY_4",	"PAY_5",	"PAY_6","pay_diff1","pay_diff2","pay_diff3","pay_diff4","pay_diff5","AMT_diff1","AMT_diff2","AMT_diff3","AMT_diff4",	"AMT_diff5","default payment next month"]
#cols = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",	"PAY_0",	"PAY_2",	"PAY_3",	"PAY_4",	"PAY_5",	"PAY_6",	"AMT_diff1","AMT_diff2","AMT_diff3","AMT_diff4",	"AMT_diff5",	"PAY_AMT1",	"PAY_AMT2",	"PAY_AMT3",	"PAY_AMT4",	"PAY_AMT5",	"PAY_AMT6",	"default payment next month"]
cols = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",	"pay_diff1","pay_diff2","pay_diff3","pay_diff4","pay_diff5",		"AMT_diff1","AMT_diff2","AMT_diff3","AMT_diff4",	"AMT_diff5",	"PAY_AMT1",	"PAY_AMT2",	"PAY_AMT3",	"PAY_AMT4",	"PAY_AMT5",	"PAY_AMT6",	"default.payment.next.month"]

if "Defualt_status" and "sum_pay"in credit_data.columns:
  values = credit_data.drop(columns=["Defualt_status", "sum_pay"], axis=1)[cols ]
else:
  values = credit_data[cols ]
print(values)
# Now impute it
scaler = MinMaxScaler(feature_range=(0, 1))
normalizedData = scaler.fit_transform(values)
credit_data = pd.DataFrame(normalizedData, columns=cols)
credit_data.head(30)


# In[31]:


plt.subplots(figsize=(25,17))
sns.heatmap(credit_data.corr(), annot=True)
plt.show()


# In[32]:


print(values.shape)


# In[34]:


X = credit_data[cols].drop(columns=["default.payment.next.month"])
y = credit_data["default.payment.next.month"]

print(y.head())


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,stratify=y)
print(type(y_train))


# ## Undersampling

# In[44]:


'''from imblearn.under_sampling import NearMiss
ns =NearMiss

X_train, y_train = ns.fit_sample(X_train, y_train)
print(pd.DataFrame(y_train).value_counts())
X_test, y_test = ns.fit_sample(X_test, y_test)
print(pd.DataFrame(y_test).value_counts())'''


# # Modelling
# 

# ## Baseline Model

# In[45]:


dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(X_train,y_train)
baseline_pred = dummy_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test,baseline_pred)
print("Baseline Accuracy is:", baseline_accuracy)


# ## Logistic Regression

# In[46]:


logReg = LogisticRegression()
logReg.fit(X_train, y_train)
y_pred = logReg.predict(X_test)


# In[48]:


lr_brforetune_cv_scores = cross_val_score(estimator=logReg, X= X_train, y= y_train, cv=10, n_jobs=-1)
mean_lr_brforetune_cv_score = lr_brforetune_cv_scores.mean()
print("The cross-validation accuracy score for untuned logistic regression after a 10 fold cross validation:", mean_lr_brforetune_cv_score)


# In[49]:


print(logReg.get_params())


# In[50]:


grid={"C":np.logspace(-20,20,20)  , 'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]}
grid_lr=GridSearchCV(logReg,grid,cv=10, n_jobs=-1)
grid_lr.fit(X_train,y_train)


# In[53]:


best_hyperparams = grid_lr.best_params
print("Best hyperparameters: \n", best_hyperparams )
mean_lr_aftertune_cv_score = grid_lr.best_score_
print("The cross-validation accuracy score for tuned logistic regression after a 10 fold cross validation:\n", mean_lr_aftertune_cv_score )
best_lr_model = grid_lr.best_estimator


# In[55]:


best_lr_fit_for_train_data = cross_val_score(estimator=best_lr_model, X=X_train, y=y_train, cv=10, n_jobs=-1).mean()
print(best_lr_fit_for_train_data)
best_lr_fit_for_test_data = cross_val_score(estimator=best_lr_model, X=X_test, y=y_test, cv=10, n_jobs=-1).mean()
print(best_lr_fit_for_test_data)


# ## Decision Tree

# In[57]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
#dt_predict = dt.predict(X_test)
#accuracy_score(y_test,dt_predict)


# In[58]:


# Cross Validation for the model
dt_brforetune_cv_scores  = cross_val_score(estimator=dt, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean_dt_brforetune_cv_score = dt_brforetune_cv_scores.mean()
print("The cross-validation accuracy score for untuned decisionTree model after a 10 fold cross validation:",  mean_dt_brforetune_cv_score)


# In[ ]:


# HyperParameter Tuning
print(dt.get_params())
params_dt = {
            'max_depth': [100,600],
            'min_samples_leaf':[50,100,1000], #100/1000/5000
            'max_features': [4,5, 6] #4/6/5,
             }
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring="accuracy", cv=10, n_jobs=-1)
grid_dt.fit(X_train,y_train)


# In[ ]:


best_hyperparams = grid_dt.best_params_
print("Best hyperparameters: \n", best_hyperparams )
dt_aftertune_cv_scores = grid_dt.best_score_
print("The cross-validation accuracy score for tuned decisionTree model after a 10 fold cross validation:\n",  dt_aftertune_cv_scores)
best_dt_model = grid_dt.best_estimator_


# In[ ]:


best_dt_fit_for_train_data = cross_val_score(estimator=best_dt_model, X=X_train, y=y_train, cv=10, n_jobs=-1).mean()
print(best_dt_fit_for_train_data)
best_dt_fit_for_test_data = cross_val_score(estimator=best_dt_model, X=X_test, y=y_test, cv=10, n_jobs=-1).mean()
print(best_dt_fit_for_test_data)


# ## Random Forest

# In[ ]:


print(X_train)


# In[ ]:


# Random Forest model
# Enter your code here
rf_model = RandomForestClassifier(n_estimators = 10000, max_features = 10, random_state = 42) 
rf_model.fit(X_train,y_train)

#print(classification_report(y_test,rf_pred))


# In[ ]:


# 10 fold cross validation
rf_brforetune_cv_scores = cross_val_score(estimator= rf_model, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean_rf_brforetune_cv_score =rf_brforetune_cv_scores.mean()
print("The cross-validation accuracy score for untuned Random Forest model after a 10 fold cross validation:", mean_rf_brforetune_cv_score)


# In[ ]:


rf_model.get_params().keys()


# In[ ]:


print(rf_model.get_params())
params_rf ={
          'n_estimators': [100,300,400,500, 600,1000],
          'criterion': ["gini", "entropy"],
          'max_depth': [100,200, 300, 400, 1000],
          'max_features': ["log2", "sqrt"],
          'bootstrap':[True, False]
}

grid_rf = GridSearchCV(estimator=rf_model, param_grid=params_rf ,cv = 3, scoring="neg_mean_squared_error", verbose = 1, n_jobs=-1)
grid_rf.fit(X_train,y_train)


# In[ ]:


best_hyperparams = grid_rf.best_params_
print("Best hyperparameters: \n", best_hyperparams )
best_rf = grid_rf.best_estimator_


# In[ ]:


best_rf_fit_for_train_data = cross_val_score(estimator=best_rf, X=X_train, y=y_train, cv=10, n_jobs=-1).mean()
print(best_rf_fit_for_train_data)
best_rf_fit_for_test_data = cross_val_score(estimator=best_rf, X=X_test, y=y_test, cv=10, n_jobs=-1).mean()
print(best_rf_fit_for_test_data)


# In[ ]:


matrix = confusion_matrix(y_test, best_rf.predict(X_test))
plt.figure(figsize = (10,7))
sns.heatmap(matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title("best rf model confusion matrix")
plt.show()


# In[ ]:


# Top 3 features for RandomForest
# Enter your code here

important_features = best_rf.feature_importances_
features_name = X.columns
imp_feature_frame = pd.DataFrame({'features': features_name, 'importance': important_features}).sort_values(by = "importance", ascending =False)
print(imp_feature_frame)
print(list(imp_feature_frame.features))


# ## AdaBoosting

# In[ ]:


base_dt = best_dt_model
ada_boost_model = AdaBoostClassifier(base_estimator= base_dt, n_estimators=200, random_state=42, learning_rate=.05)
ada_boost_model.fit(X_train, y_train)
ada_predict = ada_boost_model.predict(X_test)
print("AdaBoost Classification's Accuracy is:", accuracy_score(y_test, ada_predict))


# In[ ]:


print(ada_boost_model.get_params())


# In[ ]:


grid_ada = {'n_estimators':[100,300,500,600],
       'learning_rate':[0.01,0.08, 0.1, 0.5, 1.0]}
grid_ada = GridSearchCV(estimator=ada_boost_model, param_grid=grid_ada, scoring="accuracy", cv=10, n_jobs=-1)
grid_ada.fit(X_train,y_train)


# In[ ]:


best_hyperparams = grid_ada.best_params_
print("Best hyperparameters: \n", best_hyperparams )
ada_aftertune_cv_scores = grid_ada.best_score_
print("The cross-validation accuracy score for tuned decisionTree model after a 10 fold cross validation:\n",  ada_aftertune_cv_scores)
best_ada_model = grid_ada.best_estimator_


# In[ ]:


best_ada_fit_for_train_data = cross_val_score(estimator=best_ada_model, X=X_train, y=y_train, cv=10, n_jobs=-1).mean()
print(best_ada_fit_for_train_data)
best_ada_fit_for_test_data = cross_val_score(estimator=best_ada_model, X=X_test, y=y_test, cv=10, n_jobs=-1).mean()
print(best_ada_fit_for_test_data)


# In[ ]:


matrix = confusion_matrix(y_test, best_ada_model.predict(X_test))
plt.figure(figsize = (10,7))
sns.heatmap(matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title("best adaboost model confusion matrix")
plt.show()


# ## Accuracy Ranking
# 

# In[ ]:


dict_accuracy = {"Model":["AdaBoost Classifier","Random Forest Classifier", "Decision Tree Classifier", "Logistic Regression","Baseline Model"],
                 "Accuracy for train data": [best_ada_fit_for_train_data ,best_rf_fit_for_train_data,best_dt_fit_for_train_data, best_lr_fit_for_train_data, baseline_accuracy ],
                 "Accuracy for test data": [best_ada_fit_for_test_data ,best_rf_fit_for_test_data,  best_dt_fit_for_test_data, best_lr_fit_for_test_data, baseline_accuracy ]}
#baseline_accuracy
#mean_lr_aftertune_cv_score
#dt_aftertune_cv_scores 
#bag_aftertune_bag_cv_scores
#mean_rf_score 
#ada_aftertune_cv_scores
#mean_voting_score
#neural_aftertuned_cv_score
#"General Model cv Accuracy":[ada_aftertune_cv_scores,mean_rf_score,dt_aftertune_cv_scores ,mean_lr_aftertune_cv_score,baseline_accuracy],
df_acc = pd.DataFrame(dict_accuracy).sort_values(by="Accuracy for test data",ascending= False)

df_acc


# In[ ]:


# This is Creditor Number 16 whose payment is default 
best_rf.predict([[0.010101,	0.0,	0.166667,	0.666667,	0.051724,	0.7,	0.4,	0.555556,	0.555556,	0.636364,	0.507447,	0.766693,	0.217547,	0.530097,	0.479696,	0.003663,	0.000000,	0.001674,	0.000000,	0.003868,	0.000000	]])

