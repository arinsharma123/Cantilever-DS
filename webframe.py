from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from gensim.models import Word2Vec
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

app = Flask(__name__)


# Load the models
with open(r'pickle models\\lstm_model.pkl', 'rb') as f:
    lstm_model = pickle.load(f)

with open(r'pickle models\\word2vec_items_model.pkl', 'rb') as f:
    w2v_itemsearch_model = pickle.load(f)

with open(r'pickle models\\tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
########################################################################

## This section contains imports and functions related to SEARCH QUERY
df = pd.read_csv(r'datasets\\Preprocessed_amazonScrape.csv')
# importing functions related to search query model

df_vectors = pd.read_csv(r'datasets\\amazon_product_vectors.csv')
df_vectors['vector'] = df_vectors['vector'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
# this function preprocesses tinput query
def model_preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens

# this function will create the svg vector for the search query
def document_vector(tokens, model):
    vec = np.zeros(model.vector_size)
    count = 0
    for word in tokens:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count != 0:
        vec /= count
    return vec

# this function returns the index of the top 10 matching items as per the avg w2v model vector
def recommend_products(search_query, df, df_vectors, model):
    search_tokens = model_preprocess(search_query)
    search_vector = document_vector(search_tokens, model)
    df_vectors['similarity'] = df_vectors['vector'].apply(lambda x: cosine_similarity([search_vector], [x])[0][0])
    top_products = df_vectors.sort_values(by='similarity', ascending=False).head(10).index
    temp =  df.loc[top_products]
    temp['positive_count'] = temp['positive_feedback'].apply(lambda x: len(eval(x)))
    temp['critical_count'] = temp['critical_feedback'].apply(lambda x: len(eval(x)))
    temp['positive_feedback'] = temp['positive_feedback'].apply(lambda x: eval(x))
    temp['critical_feedback'] = temp['critical_feedback'].apply(lambda x: eval(x))
    return temp
########################################################################################

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Results page
@app.route('/results', methods=['POST'])
def results():
    query = request.form['search']
    top_items = recommend_products(query,df,df_vectors,model=w2v_itemsearch_model)
    top_items_list = top_items.to_dict(orient='records')  ##Convert DataFrame to List of Dictionaries:
    return render_template('results.html', items=top_items_list)

@app.route('/data_viz')
def data_viz():
    return render_template('data_viz.html')

if __name__ == '__main__':
    app.run(debug=True)
