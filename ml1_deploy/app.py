from codecs import latin_1_decode
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

preprocess = pickle.load(open('pipe.pkl', 'rb'))
model = tf.keras.models.load_model('model_seq_churn.h5')

st.header('Customer Probability for Churn Status')

tenure = st.number_input('Customer tenure')
monthlycharges = st.number_input('Monthly Charges')
totalcharges = st.number_input('Total Charges')

partner = st.selectbox('Do you have partner or no?', ['Yes', 'No'])
dependents = st.selectbox('Do you have dependents or no?', ['Yes', 'No'])
phoneservice = st.selectbox('Customer subscribes to home phone service with the company', ['Yes', 'No'])

multiplelines = st.selectbox('Customer subscribes to multiple telephone lines with the company', ['Yes', 'No phone service', 'No'])
internetservice = st.selectbox('Customer subscribes to Internet service with the company', ['No', 'DSL', 'Fiber Optic'])
onlinesecurity = st.selectbox('Customer subscribes to an additional online security service provided by the company', 
['No internet service', 'No' ,'Yes'])

onlinebackup = st.selectbox('Customer subscribes to an additional online backup service provided by the company',
 ['Yes', 'No', 'No internet service'])

deviceprotection = st.selectbox('Customer subscribes to an additional device protection plan for their Internet equipment provided by the company',
 ['Yes', 'No', 'No internet service'])

techsupport = st.selectbox('Customer subscribes to an additional technical support plan from the company with reduced wait times',
 ['No internet service', 'No','Yes'])

streamingtv = st.selectbox('customer uses their Internet service to stream television programing from a third party provider',
 ['No internet service', 'Yes', 'No'])
streamingmovies = st.selectbox('Customer uses their Internet service to stream television programing from a third party provider',
['No','Yes','No internet service'])
contract = st.selectbox('Customer’s current contract type',
['Month-to-month','One year','Two year'])

paperlessbilling = st.selectbox('Customer has chosen paperless billing',
['Yes','No'])
paymentmethod = st.selectbox('How the customer pays their bill',
['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'])
seniorcitizen = st.selectbox('Customer is 65 or older',
['1', '0'])

if st.button('submit'):
    
    feature_num = ['tenure', 'monthlycharges', 'totalcharges']
    feature_nonum = ['seniorcitizen']
    feature_cat = ['partner', 'dependents', 'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity', 
                    'onlinebackup', 'deviceprotection', 'techsupport','streamingtv', 'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod']

    num_df = pd.DataFrame([[tenure, monthlycharges, totalcharges]], columns=feature_num)
    no_df  = pd.DataFrame([[seniorcitizen]], columns=feature_nonum)
    cat_df = pd.DataFrame([[partner, dependents, phoneservice, multiplelines, internetservice, onlinesecurity, onlinebackup, deviceprotection, techsupport,
                            streamingtv, streamingmovies, contract, paperlessbilling, paymentmethod]], columns=feature_cat)

    X = pd.concat([num_df, no_df, cat_df], axis=1)

    trans_X = pd.DataFrame(preprocess.transform(X))
  
    pred = model.predict(trans_X)
    
    if pred[0][0] <= 0.5:
        st.text('Thank you for being a esteemed customer, \n we always look forward for you suggestion, please contact our CS HACKTVI8: 081234567890')
    else:
        st.text('Thank you for using our internet provider, \n we are sad to see you go, \n please give us review for our improvement, CS HACKTIV8: 081234567890')

  
