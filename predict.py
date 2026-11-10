import pickle
import numpy as np

import xgboost as xgb

import pandas as pd

from fastapi import FastAPI

from pathlib import Path
xgb_file = Path(__file__).parent / "xgb_model.bin"

with open(xgb_file, "rb") as f_in:
    pp, dv, model = pickle.load(f_in)

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

def predict_xgboost(df_val, pp, dv, model):
    df_val = data_preparing_predict(df_val, pp)

    columns = df_val.columns.to_list()
    dicts = df_val[columns].to_dict(orient='records')
    X_val = dv.transform(dicts)

    DX_val = xgb.DMatrix(X_val, feature_names = dv.get_feature_names_out().tolist())

    y_pred = model.predict(DX_val)

    return(y_pred)


# app itself
app = FastAPI()

@app.post("/predict")
def make_prediction(data: dict):
    predict = data
    df_pred = pd.DataFrame([predict])
    y_pred = predict_xgboost(df_pred, pp, dv, model)
    result = int(y_pred[0] > 0.5)
    return {"prediction": result}

# start the app: uvicorn predict:app --reload
# get the page: http://127.0.0.1:8000/docs

# test_p_1 = {
# "person_age": 25,
# "person_income": 8000,
# "person_home_ownership": "RENT",
# "person_emp_length": 3.0,
# "loan_intent": "MEDICAL",
# "loan_grade": "A",
# "loan_amnt": 1000,
# "loan_int_rate": 5.5,
# "loan_percent_income": 0.1,
# "cb_person_default_on_file": "N",
# "cb_person_cred_hist_length": 3
# }

# test_p_2 = {
# "person_age": 25,
# "person_income": 4000,
# "person_home_ownership": "RENT",
# "person_emp_length": 3.0,
# "loan_intent": "MEDICAL",
# "loan_grade": "E",
# "loan_amnt": 15000,
# "loan_int_rate": 15,
# "loan_percent_income": 0.1,
# "cb_person_default_on_file": "Y",
# "cb_person_cred_hist_length": 1
# }