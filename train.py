# File structure
# training the model
# saving it into a file

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

import pickle

# parameters for xgb model
xgb_eta = 0.5
xgb_max_depth = 10
xgb_min_child_weight = 5
xgb_nthread = 12

# file for saving model
output_file = "xgb_model.bin"

# paste path to your file here
df = pd.read_csv("D:\\VSCode\\AI\\Learning\\MLZoomcamp\\Midterm project\\credit_risk_dataset.csv")

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

print("length of train + validation:", df_full_train.shape[0], 
      "\nlength of validation:", len(df_val),
      "\nlength of test:", len(df_test))

# reset indices in all data
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)

# get y-s
y_train = df_train["loan_status"]
y_val = df_val["loan_status"]
y_test = df_test["loan_status"]
y_full_train = df_full_train["loan_status"]

# delete y-s from X
del df_train["loan_status"]
del df_val["loan_status"]
del df_test["loan_status"]
del df_full_train["loan_status"]

print("Data splitting finished")

def data_preparing_train(df):
    df = df.copy()
    # After the EDA, i made several decisions:
    # 
    # 1) for person_age over 100, replace with median
    # 2) change person_income by logarithm it
    # 3) for person_emp_length over 60, replace with median
    # 4) for NaN in loan_int_rate, replace with median

    # all parameters will be stored in list 
    pp = {} # pp stays for 'preparation parameters'
    # 1)
    pp["person_age_median"] = df["person_age"].median()
    df.loc[df["person_age"] > 100, "person_age"] = pp["person_age_median"]

    # 2)
    df["person_income"] = np.log(df["person_income"])

    # 3)
    pp["person_emp_length_median"] = df["person_emp_length"].median()
    df.loc[df["person_emp_length"] > 60, "person_emp_length"] = pp["person_emp_length_median"]

    # 4)
    pp["loan_int_rate_median"] = df["loan_int_rate"].median()
    df["loan_int_rate"] = df["loan_int_rate"].fillna(pp["loan_int_rate_median"])

    return(df, pp)

def data_preparing_predict(df, pp):
    df = df.copy()
    # 1)
    df.loc[df["person_age"] > 100, "person_age"] = pp["person_age_median"]

    # 2)
    df["person_income"] = np.log(df["person_income"])

    # 3)
    df.loc[df["person_emp_length"] > 60, "person_emp_length"] = pp["person_emp_length_median"]

    # 4)
    df["loan_int_rate"] = df["loan_int_rate"].fillna(pp["loan_int_rate_median"])

    
    return(df)

def train_xgboost(df_train, y_train, eta, max_depth, min_child_weight, nthread):
    df_train, pp = data_preparing_train(df_train)

    # fit the DictVectorizer
    # get column names
    columns = df_train.columns.to_list()
    dicts = df_train[columns].to_dict(orient='records')
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(dicts)

    DX_train = xgb.DMatrix(X_train, label = y_train, feature_names = dv.get_feature_names_out().tolist())

    xgb_params = {
    'eta': eta,
    'max_depth': max_depth,
    'min_child_weight': min_child_weight,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': nthread,
    'seed': 1,
    'verbosity': 1,
    }

    model = xgb.train(xgb_params, DX_train)

    return dv, model, pp


def predict_xgboost(df_val, pp, dv, model):
    df_val = data_preparing_predict(df_val, pp)

    columns = df_val.columns.to_list()
    dicts = df_val[columns].to_dict(orient='records')
    X_val = dv.transform(dicts)

    DX_val = xgb.DMatrix(X_val, feature_names = dv.get_feature_names_out().tolist())

    y_pred = model.predict(DX_val)

    return(y_pred)

print("Model training started")
dict_vectorizer, model, pp = train_xgboost(df_full_train, y_full_train, xgb_eta, xgb_max_depth, xgb_min_child_weight, xgb_nthread)
print("Model training finished")
y_pred = predict_xgboost(df_test, pp, dict_vectorizer, model)
auc = roc_auc_score(y_test, y_pred)
print(auc)

print("Model saving started")
# Saving model, pp, dv into a file
with open(output_file, "wb") as f_out:
    pickle.dump((pp, dict_vectorizer, model), f_out)

print("Model saving finished")