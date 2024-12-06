import pickle
import pandas as pd

# Load the saved Random Forest model
with open('loan_approval_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)