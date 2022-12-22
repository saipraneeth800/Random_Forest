

# https://www.kaggle.com/code/sunaysawant/diabetes-eda-logistic-random-forest/notebook


# import time
import streamlit as st
import pickle
import numpy as np


st.title("DIABETES PREDICTION")

with st.form(key='my_form'):
    name = st.text_input(label='Enter Your Name')
    age = st.number_input(label='Age')
    pregnancies = st.number_input(label='No. Of Pregnancies')
    glucose = st.number_input(label='Gluocose level')
    bp = st.number_input(label='Blood Pressure')
    thickness = st.number_input(label='Skin Thickness')
    insulin = st.number_input(label='Insulin level')
    bmi = st.number_input(label='BMI')
    dpf = st.number_input(label='Diabetes Pedigree Function')
    submit_button = st.form_submit_button(
        label='Submit')


with open('model', 'rb') as file:
    model = pickle.load(file)

values = [[pregnancies, glucose, bp, thickness, insulin, bmi, dpf, age]]
if submit_button:
    with st.spinner('Wait for it...'):
        trial = np.array(values)
        predictions = model.predict(trial)
    if predictions == [0]:
        st.success('You are not diagnosed with diabetes')
    else:
        st.warning('You are diabetic')
        st.write("Please consult doctor !")
