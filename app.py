import pickle
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕️")

diabetes_path=r"diabestes.sav"
diabetes=pickle.load(open(diabetes_path,'rb'))

st.title("DP using ML")

col1,col2,col3=st.columns(3)

with col1:
    Pregnancies=st.text_input('Number of Pregnancies')

with col2:
    Glucose=st.text_input('Glucose')

with col3:
    BloodPressure=st.text_input('Blood Pressure')

with col1:
    SkinThickness=st.text_input('Skin Thickness')

with col2:
    Insulin=st.text_input('Insulin Level')

with col3:
    BMI=st.text_input('BMI Index')

with col1:
    Age=st.text_input('Age')

with col2:
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction Level')

    diab_diagnosis=''


if st.button('Diabetes Test Result'):
    try:
        # Convert input to float
        user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), 
                      float(Insulin), float(BMI),float(Age),float(DiabetesPedigreeFunction)]
        
        # Make prediction
        diab_prediction = d_model.predict([user_input])

        # Display result
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

        st.success(diab_diagnosis)
    
    except ValueError:
        st.error("Please enter valid numerical values for all fields.")

if st.button('Show Model Accuracy'):
        
        diabetes_dataset = pd.read_csv(r"diabetes.csv")

        X_test = diabetes_dataset.drop(columns=["Outcome"])
        y_test = diabetes_dataset["Outcome"]


        y_pred = d_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy*100:.2f}%")
