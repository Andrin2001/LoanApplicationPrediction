# Loan Approval Prediction Model

## Project Overview
This repository contains a machine learning model for predicting loan approvals. The model has been trained on a dataset containing various demographic and financial features of loan applicants, sourced from the **FFIEC Dynamic National Loan-Level Dataset**. The dataset initially contained over a million observations but was reduced to 356,000 after cleaning. The model is built using a Random Forest Classifier. This README provides an overview of the **model code** and its implementation. Additionally, it includes a user-friendly application code to demonstrate the practical use of the trained model for real-world predictions.

## Project Structure
The repository contains the following files:
- **`ML_loan_data_raw.csv`**: The raw dataset used for training the loan approval model.
- **`ML_Model_LoanApplication.py`**: This Python script contains the code used to train the machine learning model for loan approval prediction. The code includes data preparation, model training, hyperparameter tuning, and saving the model for later use.
- **`ML_Prediction_LoanApplication.py`**: This script provides a user-friendly interface using **Tkinter** to input loan-related details and predict loan approval using the pre-trained model. The application validates inputs, ensures smooth navigation with helpful prompts, and presents the prediction result (Approved or Denied) in a user-friendly way.
- **`loan_approval_model.pkl`**: This is the saved version of the trained model. You can use this file to perform predictions without re-running the training process (Note: Because of the Size of the model, it is stored here: [Loan Approval Model Download](https://drive.google.com/drive/folders/153e2WTSyO3oGWIFiPiRa0JXU1-5CJiU4?usp=drive_link).
- **`requirements.txt`**: This file contains the list of dependencies required for running the project. Install these dependencies using `pip install -r requirements.txt`.

## Model Code Explanation
### Overview of `ML_Model_LoanApplication.py`
The script in `ML_Model_LoanApplication.py` is designed to train a machine learning model for predicting whether a loan application will be approved or not. Here is a breakdown of what the script does:

1. **Data Cleaning and Preparation**: The dataset is cleaned, missing values are handled, and categorical variables are converted into dummy variables to prepare the data for modeling.

2. **Model Training**: The model used is a **Random Forest Classifier**, which is effective for tabular data. Hyperparameter tuning with **Optuna** is performed to enhance the model's performance.

3. **Feature Selection and Final Model**: Features are ranked by importance to reduce overfitting and improve efficiency. The final model, trained on the selected features, achieved an Out-of-Bag (OOB) accuracy of approximately 88%. It is saved as a `.pkl` file for reuse.

## Using the Model
If you want to use the model without training it yourself, you can download the saved model from the following Google Drive link: [Loan Approval Model Download](https://drive.google.com/drive/folders/153e2WTSyO3oGWIFiPiRa0JXU1-5CJiU4?usp=drive_link).

### Application Code
The **`ML_Prediction_LoanApplication.py`** script loads the saved model and provides a graphical user interface (GUI) to collect user inputs. Key highlights of the application:
- Built using **Tkinter**, it prompts the user with a sequence of loan-related questions.
- Validates the user inputs, ensuring numerical ranges and required formats are met.
- Automatically handles missing features where applicable.
- Displays the final loan approval result (either **Loan Approved** or **Loan Denied**) in a user-friendly message box.

To use the application:
```sh
python ML_Prediction_LoanApplication.py
```

## How to Run the Model Code
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/Andrin2001/LoanApplicationPrediction.git
   ```
2. **Navigate to the Project Directory**:
   ```sh
   cd LoanApplicationPrediction
   ```
3. **Install Dependencies**:
   You will need Python and several libraries. You can install the dependencies using:
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the Model Script**:
   ```sh
   python ML_Model_LoanApplication.py
   ```
   This will load the dataset, train the model, and save the model as `loan_approval_model.pkl`. Note that this process might take some time depending on your machine's resources.
   You can also skip this and use the Model which is already saved on Google Drive.

5. **Run the Application Code**:
   ```sh
   python ML_Prediction_LoanApplication.py
   ```


