import streamlit as st
import os.path
def set_bg_hack_url():
    st.markdown(
          f"""
          <style>
          .stApp {{
              background: url("https://i.ibb.co/pwVPLh9/BK.jpg");
              background-size: cover
          }}
          </style>
          """,
          unsafe_allow_html=True
      )
set_bg_hack_url()
st.title("Welcome To Justice")

import pickle
import bz2
sfile1 = bz2.BZ2File('All Model', 'r')
models=pickle.load(sfile1)
sfile2 = bz2.BZ2File('All Vector', 'r')
vectorizer=pickle.load(sfile2)
import nltk
nltk.download('stopwords')
texts=str(st.text_input("Enter Case Study"))
import pandas as pd
df=pd.DataFrame({"case study":[texts]})


def remove_stopwords(text):
    stopwords=nltk.corpus.stopwords.words('english')
    clean_text=' '.join([word for word in text.split() if word not in stopwords])
    return clean_text
from nltk.stem.porter import PorterStemmer
def cleanup_data(df):
    # remove handle
    df['clean'] = df["case study"].str.replace("@", "") 
    # remove links
    df['clean'] = df['clean'].str.replace(r"http\S+", "") 
    # remove punctuations and special characters
    df['clean'] = df['clean'].str.replace("[^a-zA-Z]", " ") 
    # remove stop words
    df['clean'] = df['clean'].apply(lambda text : remove_stopwords(text.lower()))
    # split text and tokenize
    df['clean'] = df['clean'].apply(lambda x: x.split())
    # let's apply stemmer
    stemmer = PorterStemmer()
    df['clean'] = df['clean'].apply(lambda x: [stemmer.stem(i) for i in x])
    # stitch back words
    df['clean'] = df['clean'].apply(lambda x: ' '.join([w for w in x]))
    # remove small words
    df['clean'] = df['clean'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

names = ["K-Nearest Neighbors", "Liner SVM",
         "Decision Tree", "Random Forest",
         "ExtraTreesClassifier"]
classifier=st.selectbox("Select ML",names)                  
if st.button('Predict Section'):
    cleanup_data(df)
    feature=vectorizer.transform([df["clean"][0]])
    if classifier==names[0]:
        st.success("The Section is "+str(models[0].predict(feature)[0]))
    if classifier==names[1]:
        st.success("The Section is "+str(models[1].predict(feature)[0]))
    if classifier==names[2]:
        st.success("The Section is "+str(models[2].predict(feature)[0]))
    if classifier==names[3]:
        st.success("The Section is "+str(models[3].predict(feature)[0]))
    if classifier==names[4]:
        st.success("The Section is "+str(models[4].predict(feature)[0]))