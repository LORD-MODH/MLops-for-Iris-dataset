import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("models/iris_model.pkl")

# Title of the app
st.title("Iris Classification App")
st.write("Enter flower measurements to predict the species.")

# Input fields for features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # Make prediction
    prediction = model.predict(input_data)
    # Map prediction to species name
    species = ["Setosa", "Versicolor", "Virginica"]
    st.write(f"Predicted Species: **{species[prediction[0]]}**")

# Add your GitHub and LinkedIn links
st.markdown("---")
st.write("Developed by **Prahmodh S R**")
st.write("Connect with me:")
st.markdown(
    """
    - [GitHub](https://github.com/LORD-MODH)
    - [LinkedIn](https://www.linkedin.com/in/prahmodh-s-r/)
    """
)