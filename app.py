from flask import Flask
from flask import request, render_template
import pandas as pd
import joblib
import numpy as np

import re
import nltk
import spacy
from transformers import BertModel, BertTokenizer

import torch

nlp = spacy.load('en_core_web_md')

#Importamos las palabras de parada en inglés
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

#Agregamos emojis a las palabras de parada
newStopWords =[':D', ':d', 'd:', ";D","XD","DX","Xd","(:",":(",":/", "/:","(X","):",":B","dX",":b","b:","X)",":p","p:",":q","q:","D:","D;","W:",":W"]
stop_words.extend(newStopWords)

# Se inicializan las herramientas de bert que necesitamos para hacer la vectorización
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

classifier = joblib.load('classifier.pkl')

def limpiar(texto):
    texto = texto.lower() # Texto en minusculas

    texto = re.sub(r'@(.*?)\s+', '', texto) # Quitamos nombres de usuario
    texto = re.sub(r'http(.*?)\s+', '', texto) # Quitamos urls
    texto = re.sub(r'\d+', '', texto) # Quitamos numeros
    texto = re.sub(r'[\n\t\r]', '', texto) # Quitamos saltos de linea, tabulaciones y retornos

    # Lemmatizamos
    texto = nlp(texto)
    texto = [word.lemma_ for word in texto]
    # Eliminamos palabras de parada
    texto = ' '.join([ word for word in texto if word not in stop_words ])

    texto = re.sub(r'[^\w\s]', '', texto) # Quitamos caracteres especiales (Esto tiene que ir despues de la lematización debido al los "'s")
    texto = re.sub(r'\s+', ' ', texto) # Quitamos espacios en blanco

    texto = texto.strip() # Quitamos espacios en blanco al inicio y al final

    return texto

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route("/predice", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            #Para pruebas con jsons
            #----------------------
            #json_ = request.json
            #review_get = json_["Sentiment"]

            review_get = request.form.get('review')

            clean_review = limpiar(review_get)

            review_lst = []
            review_lst.append(clean_review)

            embeddings = []

            for i in review_lst:
                # Tokenizar el documento y agregar tokens especiales
                tokens = tokenizer.tokenize(i)
                tokens = ['[CLS]'] + tokens + ['[SEP]']

                # Convertir los tokens en IDs
                ids = tokenizer.convert_tokens_to_ids(tokens)

                with torch.no_grad():
                    outputs = model(torch.tensor([ids]))[0]
                    document_embedding = np.mean(outputs.numpy()[0], axis=0)
                    embeddings.append(document_embedding)

            
            prediction = classifier.predict(embeddings)
            prediction2 = prediction.tolist()

            if prediction2 [0][0] == 1:
                return "<p>Neutral<p>"
            elif prediction2 [0][1] == 1:
                return "<p>Positivo<p>"
            elif prediction2 [0][2] == 1:
                return "<p>Negativo<p>"

            #return prediction
        except (RuntimeError, TypeError, NameError):
            pass
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(port=5000, debug=True)