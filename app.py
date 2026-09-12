import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings("ignore")

#sst.write("Hello")

def main_page():
    st.markdown("# Basic Information")
    df=pd.read_csv("diabetes_short.csv")
    st.write('First 5 Rows of Diabetes Dataset')
    st.write(df.head())
    st.write("Last 5 rows of Diabetes Dataset")
    st.write(df.tail())
    st.write("Statistical Information")
    st.write(df.describe())
    st.balloons()
    st.snow()

def page1():
    st.markdown("# Page 1")
    st.sidebar.markdown("Page 1")

def page2():
    with open("model.pkl",'rb') as pkl1:
        classifier=pickle.load(pkl1)

    with open("std_scaler_model","rb") as pkl2:
            scaler=pickle.load(pkl2)

    st.markdown("# Diabetes Prediction")
    left,right=st.columns(2)
    Pregnancies=left.number_input("Enter Number of Pregnancies as a Whole number",step=1,value=0)
    Glucose=right.number_input("Enter Glucose in Whole Number",step=1,value=0)
    BloodPressure=left.number_input("Enter Number of BloodPressure as a Whole number",step=1,value=0)
    SkinThickness=right.number_input("Enter Number of SkinThickness as a Whole number",step=1,value=0)
    Insulin=left.number_input("Enter Number of Insulin as a Whole number",step=1,value=0)
    BMI=right.number_input("Enter Number of BMI as a Decimal number",value=0)
    DiabetesPedigreeFunction=left.number_input("Enter Number of DiabetesPedigreeFunction as a Whole number",step=1,value=0)
    Age=right.number_input("Enter Number of Age as a Whole number",step=1,value=0)
    Outcome=left.number_input("Enter Number of Outcome as a Whole number",step=1,value=0)
    predict_button=st.button("Am I a Diabetic?")

    std_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    if predict_button:
        res=classifier.predict(std_data)
        if res[0]==1:
            st.write("You are Diabetic")
        else:
            st.write("You are not Diabetic")

page_names_to_function={"Main Page":main_page,
                        "Page 1":page1,
                        "Page 2":page2}

selected_page=st.sidebar.selectbox("Select a Page",page_names_to_function.keys())
page_names_to_function[selected_page]()