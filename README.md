# Loan Approval Prediction Using ANN

A Python web application that trains an Artificial Neural Network (ANN) on loan applicant data and provides a simple Flask interface to predict whether an application is likely to be approved.

## Features
- Dataset preprocessing (missing-value imputation, categorical encoding and numeric scaling)
- ANN with two hidden layers (16 and 8 neurons) and sigmoid output
- Test accuracy, confusion matrix and classification report
- Training/validation accuracy and loss graphs
- Flask form for predicting a new applicant

## Dataset
Place the loan prediction CSV dataset in this project directory and name it **`loan_data.csv`**. It should contain the columns `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, and `Loan_Status` (Y/N). This project package does not include the dataset file.


## Setup
Use Python 3.10 or 3.11 recommended. In a terminal opened inside the project folder:


```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Train the ANN
```bash
python train_model.py
```
This creates `loan_ann_model.keras`, `loan_preprocessor.pkl`, `accuracy_graph.png`, and `loss_graph.png`.

## Run the Flask application
After training finishes:
```bash
python app.py
```
Open the local URL printed in the terminal (usually `http://127.0.0.1:5000`).

## Project files
- `train_model.py` – preprocessing, ANN training, metrics and graphs
- `app.py` – Flask application and prediction endpoint
- `templates/` – HTML pages for home, applicant form and prediction result
- `requirements.txt` – Python dependencies
- `loan_data.csv` – dataset file to add locally (not included in this package)
