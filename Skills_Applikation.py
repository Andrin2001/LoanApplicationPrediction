import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox

# Load the trained model from the .pkl file
model_path = "loan_approval_model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

class LoanApprovalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Loan Approval Prediction")
        self.root.geometry("400x200")

        self.current_step = 0
        self.inputs = []
        self.questions = [
            ("Combined Loan to Value Ratio", "Enter the combined loan to value ratio:"),
            ("Debt to Income Ratio > 60%", "Is the debt to income ratio greater than 60%? (1 for Yes, 0 for No):"),
            ("Property Value", "Enter the property value:"),
            ("Income", "Enter the applicant's annual income:"),
            ("Loan Amount", "Enter the loan amount:"),
            ("Loan Term", "Enter the loan term (in years):"),
            ("Combined Loan to Value Ratio Missing", "Is the combined loan to value ratio missing? (1 for Yes, 0 for No):"),
            ("Property Value Missing", "Is the property value missing? (1 for Yes, 0 for No):"),
            ("Applicant Age < 25 or 25-34", "Is the applicant's age less than 25 or between 25-34? (1 for Yes, 0 for No):"),
            ("Debt to Income Ratio < 20%", "Is the debt to income ratio less than 20%? (1 for Yes, 0 for No):"),
            ("Applicant Age 65-74 or > 74", "Is the applicant's age between 65-74 or greater than 74? (1 for Yes, 0 for No):"),
            ("Immigrant", "Is the applicant an immigrant? (1 for Yes, 0 for No):"),
            ("Total Units", "Enter the total number of units for the property:")
        ]

        self.label = tk.Label(root, text=self.questions[self.current_step][1], wraplength=350)
        self.label.pack(pady=10)

        self.entry = tk.Entry(root)
        self.entry.pack(pady=5)

        self.button = tk.Button(root, text="Next", command=self.next_step)
        self.button.pack(pady=10)

    def next_step(self):
        try:
            value = float(self.entry.get()) if '.' in self.entry.get() else int(self.entry.get())
            self.inputs.append(value)
            self.current_step += 1

            if self.current_step < len(self.questions):
                self.label.config(text=self.questions[self.current_step][1])
                self.entry.delete(0, tk.END)
            else:
                self.make_prediction()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric value.")

    def make_prediction(self):
        # Combine all inputs into a DataFrame to match the model's expected input format
        features = pd.DataFrame([self.inputs], columns=[
            'combined_loan_to_value_ratio',
            'debt_to_income_high',
            'property_value',
            'income',
            'loan_amount',
            'loan_term',
            'combined_loan_to_value_ratio_missing',
            'property_value_missing',
            'applicant_age_young',
            'debt_to_income_low',
            'applicant_age_old',
            'immigrant',
            'total_units'
        ])

        # Make prediction
        prediction = model.predict(features)

        # Display result in a message box
        if prediction[0] == 1:
            messagebox.showinfo("Loan Approval Result", "Loan Approved")
        else:
            messagebox.showinfo("Loan Approval Result", "Loan Denied")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = LoanApprovalApp(root)
    root.mainloop()


