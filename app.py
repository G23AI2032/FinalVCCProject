from flask import Flask, request, render_template
import pandas as pd
import pickle
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve form data
        gender = int(request.form['gender'])
        married = int(request.form['married'])
        dependents = int(request.form['dependents'])
        education = int(request.form['education'])
        self_employed = int(request.form['self_employed'])
        loan_amount = float(request.form['loan_amount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = int(request.form['credit_history'])
        property_area = int(request.form['property_area'])
        total_income = float(request.form['total_income'])
        
        # Create DataFrame
        data = {
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area],
            'Total_Income': [total_income]
        }
        
        df = pd.DataFrame(data)
        with open('rfc.pkl','rb') as file:
          load_model=pickle.load(file)
        # Display the dataframe in the result page
        if (load_model.predict(df)):
            return render_template('approved.html')
        else: 
            return render_template('not_approved.html')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
