import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/iris_model.pkl")

st.title("Iris Classification App")
st.write("Enter flower measurements to predict the species.")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = ["Setosa", "Versicolor", "Virginica"]
    st.write(f"Predicted Species: **{species[prediction[0]]}**")