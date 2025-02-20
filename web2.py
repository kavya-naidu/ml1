import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Apply custom CSS styles
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
    }
    .title {
        text-align: center;
        color: navy;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: navy;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: darkblue;
        transform: scale(1.05);
    }
    .result {
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        padding: 15px;
        border-radius: 8px;
    }
    .diabetic {
        background-color: #ff4c4c;
        color: white;
    }
    .not-diabetic {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
d_model_path = r"C:\third\diabetes.sav"

try:
    d_model = pickle.load(open(d_model_path, 'rb'))
except FileNotFoundError:
    st.error("❌ Model file not found! Please check the file path.")
    st.stop()

# Page Title
st.markdown('<div class="title">Diabetes Prediction using ML</div>', unsafe_allow_html=True)
st.markdown("---")

# User Input Section
col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.number_input('Pregnancies', min_value=0, step=1)
    SkinThickness = st.number_input('Skin Thickness', min_value=0)
    BMI = st.number_input('BMI Index', min_value=0.0, format="%.1f")

with col2:
    Glucose = st.number_input('Glucose Level', min_value=0)
    Insulin = st.number_input('Insulin Level', min_value=0)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, format="%.2f")

with col3:
    BloodPressure = st.number_input('Blood Pressure', min_value=0)
    Age = st.number_input('Age', min_value=0, step=1)

# Prediction Button
diab_diagnosis = ""

if st.button('🔍 Check Diabetes Result'):
    try:
        # Prepare input data
        user_input = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        
        # Make prediction
        diab_prediction = d_model.predict(user_input)

        # Display result with styling
        if diab_prediction[0] == 1:
            diab_diagnosis = '<div class="result diabetic">The person is diabetic 😔</div>'
        else:
            diab_diagnosis = '<div class="result not-diabetic">The person is not diabetic 😊</div>'

        st.markdown(diab_diagnosis, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Sidebar - Model Accuracy Section
st.sidebar.title("⚙️ Model Info")
if st.sidebar.button("Show Model Accuracy"):
    try:
        diabetes_dataset = pd.read_csv(r"C:\third\diabetes.csv")

        # Extract features and target
        X_test = diabetes_dataset.drop(columns=["Outcome"])
        y_test = diabetes_dataset["Outcome"]

        # Predict and calculate accuracy
        y_pred = d_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.sidebar.success(f"✅ Model Accuracy: {accuracy*100:.2f}%")

    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>Developed by <b>kavya</b> P")
