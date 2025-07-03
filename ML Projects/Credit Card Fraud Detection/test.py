import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

df=pd.read_csv("creditcard.csv")

legit=df[df.Class==0]
fraud=df[df.Class==1]

legit_sample=legit.sample(n=492) # this takes random 492 values so we have equal number
                                 # s of o and 1 ie legit and fraud values this makes the balance between the data 

newdf=pd.concat([legit_sample,fraud],axis=0) # this will concat both legit sample and fraud 
newdf.shape

newdf.groupby('Class').mean() # finding the mean value of o and 1 
x=newdf.drop(columns='Class', axis=1)
y=newdf['Class']

import warnings
warnings.filterwarnings("ignore")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
model=LogisticRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)

# # Web App 
# st.title("Credit Card Fraud Detection model")
# input_df=st.text_input("Enter all require feature value")
# splited_input=input_df.split(',')

# submit=st.button("Submit")

# if submit:
#     features=np.asarray(splited_input,dtype=float)#converting input into machine understandable language that is numpy array
#     prediction=model.predict(features.reshape(1,-1)) # making prediction on featuer either 0 or 1

#     if prediction[0]==0:
#         st.write("Legitimate Transaction")
#     else : 
#         st.write("Fraud Transaction")

import numpy as np
from PIL import Image

# Page Config
st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")

# Header
st.title("💳 Credit Card Fraud Detection")
st.markdown("Predict if a transaction is **Legitimate or Fraudulent** based on input features.")

# Sidebar Info
with st.sidebar:
    st.header("📘 Instructions")
    st.markdown("""
    - Enter all feature values separated by commas.
    - Example: `0.1, 1.2, -0.3, ...` (based on your model's input).
    - Click **Submit** to get prediction.
    """)
    st.info("Ensure the input matches model feature count!")

# Input Field
input_df = st.text_input("🔢 Enter all required feature values (comma-separated):")

# Submit Button
submit = st.button("🚀 Predict")

# On Submit
if submit:
    try:
        # Convert to numpy array
        splited_input = input_df.split(',')
        features = np.asarray(splited_input, dtype=float).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Result UI
        if prediction[0] == 0:
            st.success("✅ Legitimate Transaction")
        else:
            st.error("⚠️ Fraudulent Transaction Detected!")
    except:
        st.warning("⚠️ Please enter valid numbers matching the required features.")
