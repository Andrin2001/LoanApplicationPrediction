# Loan Approval Prediction Model

## Project Overview
This repository contains a machine learning model for predicting loan approvals. The model has been trained on a dataset containing various demographic and financial features of loan applicants and is built using a Random Forest Classifier. This README provides an overview of the **model code**, while another application code file will be provided later to demonstrate the use of the model in practice.

## Project Structure
The repository contains the following files:
- **`Skills_Model.py`**: This Python script contains the code used to train the machine learning model for loan approval prediction. The code includes data preparation, model training, hyperparameter tuning, and saving the model for later use.
- **`loan_approval_model.pkl`**: This is the saved version of the trained model. You can use this file to perform predictions without re-running the training process.
- **Application Code**: To be provided later, this code will demonstrate how to use the trained model to make predictions.

## Model Code Explanation
### Overview of `Skills_Model.py`
The script in `Skills_Model.py` is designed to train a machine learning model for predicting whether a loan application will be approved or not. Here is a breakdown of what each section does:

1. **Data Cleaning and Preparation**: The code starts by importing the necessary libraries and loading the dataset. Several steps are performed to clean the data, including handling missing values, converting categorical variables into dummy variables, and normalizing the data.

2. **Training the Model**:
   - The model used in this project is a **Random Forest Classifier**. It is well-suited for tabular data and performs well with a variety of feature types.
   - **Hyperparameter tuning** is performed using **Optuna** to optimize the model and improve its predictive performance.

3. **Feature Selection**: The model also includes a feature selection step where features are ranked by their importance in the model. This helps to reduce overfitting and improve the performance by keeping only the most relevant features.

4. **Saving the Model**: The resulting model is saved in a `.pkl` file (`loan_approval_model.pkl`) using **Pickle**. This allows users to load the model later without having to retrain it, which can save considerable time.

## Using the Model
If you want to use the model without training it yourself, you can download the saved model from the following Google Drive link: [Loan Approval Model Download](https://drive.google.com/drive/folders/153e2WTSyO3oGWIFiPiRa0JXU1-5CJiU4?usp=drive_link).

### Application Code (To Be Provided Later)
The application code will demonstrate how to load the saved model (`loan_approval_model.pkl`) and use it to make predictions. It will include a simple user interface or a script that accepts applicant details as input and returns the loan approval decision.

## How to Run the Model Code
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/username/loan-approval-prediction.git
   ```
2. **Install Dependencies**:
   You will need Python and several libraries. You can install the dependencies using:
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the Model Script**:
   ```sh
   python Skills_Model.py
   ```
   This will load the dataset, train the model, and save the model as `loan_approval_model.pkl`. Note that this process might take some time depending on your machine's resources.




