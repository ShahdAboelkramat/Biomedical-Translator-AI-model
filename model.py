from transformers import MarianMTModel, MarianTokenizer 
from torchaudio.transforms import MelSpectrogram
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datasets import load_dataset
from gtts import gTTS
import torchaudio
import numpy as np
import pandas as pd
import os
import re


#load dataset

data=pd.read_csv(r"G:\\INS kher\\train\\large_diseases_dataset.csv")


#preprocessing & cleaning 

data.dropna(inplace=True) #delete unwanted raws 

data.fillna("invalid",inplace=True) #fill any empty cell with invalid 

data.drop_duplicates(inplace=True) #remove any duplicated values

def clean_text(text):
    text = text.lower()  #   
    text = re.sub(r'[^\w\s]', '', text)  # remove any special char
    return text

data['English'] = data['English'].apply(clean_text)
data['Deutsch'] = data['Deutsch'].apply(clean_text)
data['Spanish'] = data['Spanish'].apply(clean_text)
data['Portuguese'] = data['Portuguese'].apply(clean_text)


#Convert from text to numbers

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["English"]).toarray()

Y_german = vectorizer.transform(data["Deutsch"]).toarray()
Y_spanish = vectorizer.transform(data["Spanish"]).toarray()
Y_portuguese = vectorizer.transform(data["Portuguese"]).toarray()


#split the dataset for tarining and testing 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_german, test_size=0.2, random_state=42)



#taning
model = LinearRegression() #create model 

model.fit(X_train, Y_train) #train the model


#testing and calculate the error 

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")



new_text = ["Ulcer"]
new_text_vectorized = vectorizer.transform(new_text).toarray()

predicted_translation = model.predict(new_text_vectorized)


def numbers_to_text(predicted_vector, vectorizer):
    words = vectorizer.get_feature_names_out()  # الحصول على الكلمات من الـ vectorizer
    text = []

    for vector in predicted_vector:
        top_indices = np.argsort(vector)[-5:]  # أخذ أعلى 5 قيم (يمكنك تغيير العدد)
        top_words = [words[i] for i in top_indices if vector[i] > 0.01]  # اختيار الكلمات غير الصفرية
        text.append(" ".join(top_words))

    return text


predicted_text = numbers_to_text(predicted_translation, vectorizer)
print("Predicted German Translation (Vectorized):", predicted_text)




