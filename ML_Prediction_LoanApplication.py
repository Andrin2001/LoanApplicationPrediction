################################################################################
# Programming - Introduction Level
# Predicting Loan Approvals
################################################################################

# Application which predicts loan approval based on loan-related data from user inputs.
# For some parts in the code, we used ChatGPT as a supporting role and to give us ideas. 
# First, we used ChatGPT for the tkinter GUI, therefore setting up an interface suitable for the inputs (setting up buttons, frame etc.)
# Secondly, we used ChatGPT to give us an idea/ inspiration for the button implementation. 

import pickle
import pandas as pd
import tkinter as tk
from tkinter import messagebox

# Load the pre-trained loan approval model from the file
# This model is used to predict loan approval based on user inputs
model_path = "loan_approval_model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

class LoanApprovalApp:
    """
    A GUI application based on Tkinter that collects loan-related data from the user
    and predicts loan approval using a pre-trained model.
    """
    def __init__(self, root):
        """
        Initialize the LoanApprovalApp with a series of questions and a Tkinter interface.
        """

        #Set window properties
        self.root = root
        self.root.title("Loan Approval Prediction") # Set the title of the window
        self.root.geometry("500x300") # Define the size of the window

        # Keep track of the current question (step) and user inputs
        self.current_step = 0
        self.inputs = []

        # Define the questions, valid ranges, and other related data
        # Each element is a tuple: (Name, the input text, (min value and max value))
        self.questions = [
            ("Combined Loan to Value Ratio", "Enter the combined loan to value ratio:", (0, None)),
            ("Debt to Income Ratio > 60%", "Is the debt to income ratio greater than 60%? (1 for Yes, 0 for No):", (0, 1)),
            ("Property Value", "Enter the property value:", (0, None)),
            ("Income", "Enter the applicant's annual income:", (0, None)),
            ("Loan Amount", "Enter the loan amount:", (0, None)),
            ("Loan Term", "Enter the loan term (in years):", (0, 40)),
            ("Applicant Age < 25 or 25-34", "Is the applicant's age less than 25 or between 25-34? (1 for Yes, 0 for No):", (0, 1)),
            ("Debt to Income Ratio < 20%", "Is the debt to income ratio less than 20%? (1 for Yes, 0 for No):", (0, 1)),
            ("Applicant Age 65-74 or > 74", "Is the applicant's age between 65-74 or greater than 74? (1 for Yes, 0 for No):", (0, 1)),
            ("Immigrant", "Is the applicant an immigrant? (1 for Yes, 0 for No):", (0, 1)),
            ("Total Units", "Enter the total number of units for the property:", (1, None))
        ]

        # Provide extra information (help texts) for some questions
        # This explains the more complex questions or calculations
        self.help_texts = [
            "Combined Loan to Value Ratio: Ratio of the loan amount to the property value. Formula: (Loan Amount / Property Value) * 100.",
            "The Debt-to-income ratio indicates if the applicant's debts exceed 60% of their income. Formula: (Total monthly debt / Gross monthly income) * 100",
            "Property Value: The market value of the property.",
            "Income: Applicant's annual income before taxes.",
            "",
            "Loan Term: The loan repayment period in years.",
            "",
            "",
            "",
            "",
            "Total Units: Number of residential units associated with the property."
        ]

        # Setup which questions should have a help button
        self.help_enabled = [
            True,  # Combined Loan to Value Ratio
            True,  # Debt to Income Ratio > 60%
            True,  # Property Value
            True,  # Income
            False, # Loan Amount
            True,  # Loan Term
            False, # Applicant Age < 25 or 25-34 
            False,  # Debt to Income Ratio < 20%
            False, # Applicant Age 65-74 or > 74 
            False, # Immigrant
            True   # Total Units
        ]

        # Create a label to display the question text
        self.label = tk.Label(root, text=self.questions[self.current_step][1], wraplength=450)
        self.label.pack(pady=15, padx=20)

        # Create an entry field where the user can type their answer
        self.entry = tk.Entry(root)
        self.entry.pack(pady=10, padx=20)

        # Create a frame for buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=15)

        # Add buttons for "Next", "Help", and "Reset"
        self.next_button = tk.Button(button_frame, text="Next", command=self.next_step)
        self.help_button = tk.Button(button_frame, text="Help", command=self.show_help)
        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset_app)

        # Arrange the buttons side by side
        self.next_button.pack(side=tk.LEFT, padx=10)
        self.help_button.pack(side=tk.LEFT, padx=10)
        self.reset_button.pack(side=tk.LEFT, padx=10)


    def show_help(self):
        """
        Display additional information about the current question in a message box.
        """
        current_index = self.current_step
        if self.help_enabled[current_index]:  # Check if help is available for the current question
            messagebox.showinfo("Help/Info", self.help_texts[current_index])
        else:
            messagebox.showinfo("Help/Info", "No additional information available for this question.")

    def update_help_button(self):
        """
        Show or hide the Help button depending on the current question.
        """
        if self.help_enabled[self.current_step]: 
            self.help_button.config(state=tk.NORMAL)
        else:
            self.help_button.config(state=tk.DISABLED)

    def reset_app(self):
        """
        Reset the application to the first question.
        Clears all inputs and restarts the question sequence.
        """
        self.current_step = 0  # Reset the current question index
        self.inputs = []  # Clear all saved inputs
        self.label.config(text=self.questions[self.current_step][1])  # Reset question label
        self.entry.delete(0, tk.END)  # Clear the input field
        self.update_help_button() #Help button update for the first question again.

    def next_step(self):
        """
        Validate the input, save it, and move to the next question.
        """
        try:

            # Automatically set missing variable questions to 1
            if self.current_step == 5: 
                self.inputs.append(1) # For the variable combined_loan_to_value_ratio_missing, we add 1 because this has to be true in order that the model can predict.
                self.inputs.append(1) # property_value_missing variable has also to be 1 per default. 

            input_value = self.entry.get()
            if not input_value.strip():
                messagebox.showerror("Input Error", "Please enter a value before proceeding.")
                return
            

            value = float(self.entry.get()) if '.' in self.entry.get() else int(self.entry.get())
            valid_range = self.questions[self.current_step][2]
            if valid_range == (0, 1) and value not in (0, 1): #Questions with 1 or 0 as inputs
                raise ValueError("Value must be either 0 (No) or 1 (Yes).")
            if valid_range[0] is not None and value < valid_range[0]: #Questions with numbers that have to be higher than a certain number ( for example only positive numbers)
                raise ValueError(f"Value must be greater than or equal to {valid_range[0]}.")
            if valid_range[1] is not None and value > valid_range[1]: #Questions with numbers which have an upper limit (for example loan term)
                raise ValueError(f"Value must be less than or equal to {valid_range[1]}.")
        
            # Save the input and move to the next question
            self.inputs.append(value)
            self.current_step += 1

            # Special Logic:
            # Skip Question 9 (index 8) if Question 7 (index 6) is answered Yes (1).
            # This assumes that applicants younger than 34 are not considered for old-age categories.

            if self.current_step == 8 and self.inputs[6] == 1:  # Question 7 (index 6)
                self.inputs.append(0)  # Automatically set Question 9 to No (0)
                self.current_step = 9  # Skip to Question 10

            # Update interface for next question
            if self.current_step < len(self.questions):
                self.label.config(text=self.questions[self.current_step][1])
                self.entry.delete(0, tk.END)
                self.update_help_button()  # Update Help button visibility
            else:
                self.make_prediction() # If all questions answered, predict the result
        except ValueError as e:
            messagebox.showerror("Input Error", str(e)) # Show an error if the input is invalid

    def make_prediction(self):
        """
        Use the trained model to predict loan approval based on inputs.
        """
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

        # Make a prediction using the loaded model
        prediction = model.predict(features)

        # Display the result to the user
        if prediction[0] == 1:
            messagebox.showinfo("Loan Approval Result", "Loan Approved")
        else:
            messagebox.showinfo("Loan Approval Result", "Loan Denied")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = LoanApprovalApp(root)
    root.mainloop()
