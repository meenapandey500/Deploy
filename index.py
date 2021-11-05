import streamlit as st
import pickle
file1 = open("model.pkl","rb")
file2 = open("tfidf.pkl","rb")

model=pickle.load(file1)
tfidf=pickle.load(file2)

def predict_sms(msg):
    msg_tfidf=tfidf.transform([msg])
    result=model.predict(msg_tfidf)[0]
    return result

st.write(" # spam sms detector ")

sms=st.text_area("Message")

if st.button("predict"):
    result=predict_sms(sms)
    st.write(result)