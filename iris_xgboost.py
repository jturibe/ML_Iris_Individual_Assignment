import streamlit as st
import pickle
import numpy as np

# Load the model
with open("xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)

# Function to get user input
def get_user_input():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.8)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 4.5)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.5)
    return np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

# Main function to run the app
def run():
    st.title('Iris Species Prediction')
    st.write('Enter the dimensions of the iris:')
    user_input = get_user_input()
    prediction = model.predict(user_input)
    st.write(f'Prediction: {prediction}')

if __name__ == '__main__':
    run()
