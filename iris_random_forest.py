import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# Load the model
with open('iris_random_forest.pkl', 'rb') as file:
    model = pickle.load(file)

# Get user input
def get_user_input():
    user_data = pd.DataFrame({
        'sepal length (cm)': [st.session_state['sepal_length']],
        'sepal width (cm)': [st.session_state['sepal_width']],
        'petal length (cm)': [st.session_state['petal_length']],
        'petal width (cm)': [st.session_state['petal_width']]
    })
    return user_data

# Header
st.markdown("""
## Iris Flower Prediction App
Enter the dimensions of the flower to get the species.
""")

# Define target names
target_names = ['Setosa', 'Versicolor', 'Virginica']

# Make predictions and Display
if 'sepal_length' not in st.session_state:
    st.session_state['sepal_length'] = 5.4
if 'sepal_width' not in st.session_state:
    st.session_state['sepal_width'] = 3.4
if 'petal_length' not in st.session_state:
    st.session_state['petal_length'] = 1.3
if 'petal_width' not in st.session_state:
    st.session_state['petal_width'] = 0.2

user_input = get_user_input()
with st.spinner('Predicting...'):
    prediction = model.predict(user_input)

col1, col2 = st.columns(2)
with col1:
    # Display the image of the predicted class
    image = Image.open(f"{target_names[prediction[0]]}.jpg")
    st.image(image, use_column_width=True, caption=f'{target_names[prediction[0]]}')
with col2:
    st.write(f"## Your flower is a... {target_names[prediction[0]]} :blossom:")

# Separator
st.markdown("---")

# Input Parameters
st.header("Input Parameters")

# Increment and Decrement sliders with custom labels
def increment_decrement_sliders(param_name, label):
    col1, col2, col3 = st.columns([2, 1.5, 2])
    with col1:
        new_value = st.slider(label, 0.0, 10.0, st.session_state[param_name], 0.25)
    with col2:
        st.write(f'Current {label}: {new_value}')
    with col3:
        st.session_state[param_name] = new_value

increment_decrement_sliders('sepal_length', 'Sepal Length (cm)')
increment_decrement_sliders('sepal_width', 'Sepal Width (cm)')
increment_decrement_sliders('petal_length', 'Petal Length (cm)')
increment_decrement_sliders('petal_width', 'Petal Width (cm)')
