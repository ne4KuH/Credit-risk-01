# Credit-risk-01
Working on Credit scoring project. Goal - build several simple models, tune them, choose the best one. EDA included, as well as Docker deployment. Working with https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data dataset

# Description of the Problem

Credit scoring is the process of assessing the risk of a borrower failing to repay a loan. The goal of this project is to build machine learning models that predict whether a loan applicant is likely to default.

The dataset contains information about applicants, including their age, income, employment history, home ownership, loan purpose, loan amount, interest rate, and credit history.

The target variable is *loan_status*:

- 0 – The borrower does not default
- 1 – The borrower defaults

The models use these features to estimate the probability of loan default and help assess the credit risk of potential borrowers.

# Get the data

The data is located in Kaggle. You can explore and download it from [here](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data).

Alternatively, you could use cURL:
```
#!/bin/bash
curl -L -o ~/Downloads/credit-risk-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/laotse/credit-risk-dataset
```

Extract .zip file into a folder, you want your project to be 

# How to run the project

Here are three options:
- Local EDA analysis, in `main.ipynb`
- Run FastAPI application with uvicorn and test the model via HTTP POST request
- Deploy a Docker container with app inside and test the model via HTTP POST request


## Local EDA analysis

There are files `main.ipynb` and `requirements.txt`. Download them and move to the same folder, where extracted `credit_risk_dataset.csv` is. After that you can use `requirements.txt` to install all necessary packages. Then, you could freely explore `main.ipynb` and test models.

F.e. if you use VSCode with pip and cmd terminal here are steps needed:
1) Navigate to your project folder
```cmd
cd path_to_your_folder_with_project
```
2) Create virtual environment
```cmd
python -m venv .venv
```
3) Activate the environment
```cmd
.venv\Scripts\activate.bat
```
4) Install required packages
```cmd
pip install -r requirements.txt
```

You can now test `main.ipynb` file

## FastAPI app

There are files `train.py` and `predict.py`. Download them to the folder, described in previous block.

`train.py`
- feel free to change parameters and code at `train.py`
- after running `train.py` will produce a `xgb_model.bin` file, that you can use for `predict.py`

in order to run FastAPI app, use next command:
```cmd
uvicorn predict:app --reload
```
*you must be in your venv and run this command from the project folder*

If you do everything right, you will see next message:
```
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

after that you can navigate in your web browser to http://127.0.0.1:8000/docs and test the app. Simply press **Try it out** in *POST /predict* and paste json with loan record. F.e.:
```json
{
"person_age": 25,
"person_income": 8000,
"person_home_ownership": "RENT",
"person_emp_length": 3.0,
"loan_intent": "MEDICAL",
"loan_grade": "A",
"loan_amnt": 1000,
"loan_int_rate": 5.5,
"loan_percent_income": 0.1,
"cb_person_default_on_file": "N",
"cb_person_cred_hist_length": 3
}
```
After that press **Execute**. Response will be shown in *Responses* block below.

## Docker deploy

For a successful Docker deploy, you need to have next files on a separate folder:
- `Dockerfile`
- `predict.py`
- `requirements_docker.txt`
- `xgb_model.bin` you can use mine, or create your own by running `train.py` 

1) navigate to your folder
```powershell
cd .\path\to\your\folder
```
2) build the docker image
```powershell
docker build -t loan-prediction-api .
```
3) start docker container
```powershell
docker run -d -p 8000:8000 --name loan-api loan-prediction-api
```
4) check whether it is running
```powershell
docker ps
```

after that you can navigate in your web browser to http://localhost:8000/docs and test the app. Simply press **Try it out** in *POST /predict* and paste json with loan record. F.e.:
```json
{
"person_age": 25,
"person_income": 8000,
"person_home_ownership": "RENT",
"person_emp_length": 3.0,
"loan_intent": "MEDICAL",
"loan_grade": "A",
"loan_amnt": 1000,
"loan_int_rate": 5.5,
"loan_percent_income": 0.1,
"cb_person_default_on_file": "N",
"cb_person_cred_hist_length": 3
}
```
After that press **Execute**. Response will be shown in *Responses* block below.

At the end, you can stop the container
```powershell
docker stop loan-api
```
or delete it
```powershell
docker stop loan-api
docker rm loan-api
```
