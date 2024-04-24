import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open('rf_model.sav', 'rb') as file:
    model = pickle.load(file)

# Define a function to preprocess user input
def preprocess_input(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    # Perform any necessary preprocessing here, such as encoding categorical variables
    # For simplicity, we'll just return a numpy array of the input values
    return np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])

# Define a function to predict stroke probability
def predict_stroke_probability(input_data):
    # Preprocess the input data
    input_data_processed = preprocess_input(*input_data)
    # Predict the probability of stroke
    probability = model.predict_proba(input_data_processed)[0][1]
    return probability

# Define a function to interpret prediction
def interpret_prediction(probability):
    if probability >= 0.5:
        return "The result suggests a higher likelihood of stroke. Consider adopting lifestyle changes such as regular exercise, healthy diet, and smoking cessation to reduce the risk."
    else:
        return "The result indicates a low risk of stroke at the moment. It's a good time to stay vigilant with health monitoring and maintain a healthy lifestyle."

# Define a function to display a colorful bar chart
def display_bar_chart(probability):
    colors = ['green', 'red']
    labels = ['No Stroke', 'Stroke']
    data = [1 - probability, probability]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=labels, y=data, palette=colors, ax=ax)
    ax.set_xlabel('Prediction', fontsize=14)
    ax.set_ylabel('Probability', fontsize=14)
    ax.set_title('Stroke Prediction Probability', fontsize=16)
    st.pyplot(fig)

# Create a Streamlit web app
def main():
    # Set app title and description
    st.title("Stroke Predictor App")
    st.write("Enter the required information to predict the likelihood of stroke.")

    # Create input fields for user to enter numerical information
    age = st.slider("Age", 1, 100, 30)
    avg_glucose_level = st.number_input("Average Glucose Level", 0.0, 300.0, 80.0)
    bmi = st.number_input("BMI", 0.0, 100.0, 20.0)

    # Create a sidebar for file upload and categorical columns
    st.sidebar.title("Input Parameters")

    # Allow user to upload a file
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file for intput", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

    # Create input fields for user to enter information
    st.sidebar.header("Fill the Info")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.sidebar.selectbox("Ever Married ?", ["No", "Yes"])
    work_type = st.sidebar.selectbox("Select a Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Do you Smoke ?", ["Unknown", "Never smoked", "formerly smoked", "smokes"])

    # Create a button to predict stroke probability
    if st.button("Show Result"):
        # Gather input features
        input_data = (1 if gender == "Male" else 0, age, 1 if hypertension == "Yes" else 0,
                      1 if heart_disease == "Yes" else 0, 1 if ever_married == "Yes" else 0,
                      ["Private", "Self-employed", "Govt_job", "children", "Never_worked"].index(work_type),
                      1 if Residence_type == "Urban" else 0, avg_glucose_level, bmi,
                      ["Unknown", "Never smoked", "formerly smoked", "smokes"].index(smoking_status))
        # Predict stroke probability
        probability = predict_stroke_probability(input_data)

        # Interpret prediction
        prediction_statement = interpret_prediction(probability)

        # Display prediction statement and probability percentages
        st.write(prediction_statement)
        st.write(f"Stroke Probability: {probability * 100:.2f}% (likely) / {(1 - probability) * 100:.2f}% (unlikely)")

        # Display a colorful bar chart
        display_bar_chart(probability)

# Run the web app
if __name__ == "__main__":
    main()
