from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model_columns = pickle.load(open("model_columns.pkl","rb"))
model = pickle.load(open("model.pkl", "rb"))
print('model loaded')


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    
    
   # Getting the data from the form
    income1=int(request.form['ApplicantIncome'])
    income2=int(request.form['CoaplicantIncome'])
    LoanAmount=int(request.form['LoanAmount'])
    Dependents=int(request.form['Dependents'])
    Loan_Amount_Term=int(request.form['Loan_Amount_Term'])
    Credit_History=int(request.form['Credit_History'])
    Self_Employed=int(request.form['Self_Employed'])
    Property_Area=int(request.form['Property_Area'])
    Education=request.form['Education']
    Married=int(request.form['Married'])
    Gender=int(request.form['Gender'])
    
    #  creating a json object to hold the data from the form
    input_data=[{
    'income1':income1,
    'income2':income2,
    'LoanAmount':LoanAmount,
    'Loan_Amount_Term':Loan_Amount_Term,
    'Dependents': Dependents,
    'Credit_History':Credit_History,
    'Self_Employed':Self_Employed,
    'Property_Area':Property_Area,
    'Education':Education,
    'Married':Married,
    'Gender':Gender
    }]


    dataset=pd.DataFrame(input_data)
    print(dataset)
    print('Shape of dataframe', dataset.shape)
    
    #int_features = [x for x in request.form.values()]
    print('Dummification for one feature')
    query = pd.get_dummies(data=dataset,columns=['Education'],prefix_sep='-')
    print(query)
    print('After dummification :',query.shape)
    final_features = np.array(query)
    print(final_features)
    prediction = model.predict(final_features)
    output=prediction[0]
    print('prediction is ', output)
    if output==1:
        return render_template("index.html", prediction_text= " APPROVED ")
    else:
        return render_template("index.html", prediction_text= " REJECTED ")
print('prediction printed')

@app.route('/results',methods=['POST'])
def results():

    query = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])
    print(query)
    query = query.reindex(columns=model_columns, fill_value=0)
    query = pd.get_dummies(pd.DataFrame(query),columns=['Education'])
    final_features = np.array(query)
    prediction = list(model.predict(final_features))
    return jsonify({'prediction': str(prediction[0])})
    #output = prediction[0]
    #return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    
    
