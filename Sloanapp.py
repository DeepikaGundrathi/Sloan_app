import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model from the pickle file
model_filename = 'rf_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Define the app
st.title('Sloan Digital Sky Survey Object Classification')
st.write('Predict whether an object is a star, galaxy, or QSO based on its features.')

# Define the input fields for user to enter data
def user_input_features():
    ra = st.number_input('Right Ascension (ra)', min_value=0.0)
    dec = st.number_input('Declination (dec)', min_value=0.0)
    u = st.number_input('u')
    g = st.number_input('g')
    r = st.number_input('r')
    i = st.number_input('i')
    z = st.number_input('z')
    redshift = st.number_input('Redshift')
    run = st.number_input('Run', min_value=0)
    rerun = st.number_input('Rerun', min_value=0)
    camcol = st.number_input('Camcol', min_value=0)
    field = st.number_input('Field', min_value=0)
    plate = st.number_input('Plate', min_value=0)
    mjd = st.number_input('MJD', min_value=0)
    fiberid = st.number_input('Fiber ID', min_value=0)
    
    data = {
        'ra': ra,
        'dec': dec,
        'u': u,
        'g': g,
        'r': r,
        'i': i,
        'z': z,
        'redshift': redshift,
        'run': run,
        'rerun': rerun,
        'camcol': camcol,
        'field': field,
        'plate': plate,
        'mjd': mjd,
        'fiberid': fiberid
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the input data
st.subheader('User Input features')
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')

# Define the class labels based on your model's classes
class_labels = ['Galaxy', 'QSO', 'Star']

# Map prediction to class label
predicted_class_label = class_labels[prediction[0]]

st.write(predicted_class_label)

st.subheader('Prediction Probability')
st.write(prediction_proba)
