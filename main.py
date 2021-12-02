import streamlit as st
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
import re
import pdfplumber
import pickle
from nltk.corpus import stopwords 

filename = 'final_model.sav'
model = pickle.load(open(filename, 'rb'))

stopwords= stopwords.words('english')
def cleanText(text):
    text = re.sub(r'''!\(\)-\[]\{};:'"\,<>./?@#$%^&*_~''', r' ', text) 
    text = text.lower()
    text = text.replace(',', '')
    tokens = nltk.word_tokenize(text)
    wordlist = [] 
    for w in tokens:
      if w not in stopwords:
        if w.isalpha():
          wordlist.append(w)

    clean_text = ' '.join(wordlist)
    return clean_text

body = st.container()

def print_imgs(li):
    n= 1
    for i in li:
        if n==6:
            n= 1
        if n==1:
            c1.image('imgs/s'+str(i)+'.jpg')
            c1.markdown('Probability : '+str(vals[n-1]))
        elif n==2:
            c2.image('imgs/s'+str(i)+'.jpg')
            c2.markdown('Probability : '+str(vals[n-1]))
        elif n==3:
            c3.image('imgs/s'+str(i)+'.jpg')
            c3.markdown('Probability : '+str(vals[n-1]))
        elif n==4:
            c4.image('imgs/s'+str(i)+'.jpg')
            c4.markdown('Probability : '+str(vals[n-1]))
        elif n==5:
            c5.image('imgs/s'+str(i)+'.jpg')
            c5.markdown('Probability : '+str(vals[n-1]))
        n += 1

def predict(text):
    text = [text]
    prediction = model.predict_proba(text)
    prediction = prediction[0]
    sdgs = []
    for index,value in enumerate(prediction):
        if value > 0.1:
            sdgs.append(index+1)
            vals.append(round(value, 2))
    return [prediction , sdgs]

def extract_data(feed):
    data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_text())
        text = ' '.join(data)
    return text


with body:
    st.header('MDSC 302 : SDG Classifier')

    input_text = st.text_area('Drop some text here')
    sample = st.selectbox('Upload a txt file', (None,'zee_entertinment.txt', 'vedanta.txt'), index= 0)
    if sample != 'None':
        f = open("sample_CSR.txt", "r",encoding='utf-8')
        input_text = f.read()


    txt_col, pdf_col = st.columns(2)

    txt_col.subheader('upload a txt file')
    uploaded_file = txt_col.file_uploader('.txt file', type="txt")
    if uploaded_file is not None:
        input_text = str(uploaded_file.read(), 'utf-8')
    
    pdf_col.subheader('upload a PDF file')
    uploaded_file = pdf_col.file_uploader('.pdf file', type="pdf")
    if uploaded_file is not None:
        input_text = extract_data(uploaded_file)

    vals = []
    input_text = cleanText(input_text)
    prediction , nums = predict(input_text)

    if  st.button("Let's know the SDG's"):
        c1, c2, c3, c4, c5 = st.columns(5)        
        print_imgs(nums)
