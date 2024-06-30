# Amazon Web Scrape and Search Engine model
![Screenshot (182)](https://github.com/arinsharma123/Cantilever-Web-Srape/assets/128144029/1cf8aa85-01a2-4f5b-b090-1cc651435cda)
## Overview:
This project implements a full data processing and search engine pipeline, starting with web scraping for data collection. The data undergoes wrangling and thorough cleaning. An LSTM model, trained on a third-party dataset with 89% accuracy, classifies sentiment in the scraped reviews. Word2Vec embeddings are used to train a semantic search model, which provides the top 10 matching products for subjective and informal user queries. The project also includes a brief analysis of combined textual and numerical data using data visualization to understand user behavior for specific product types. The models are deployed using the Flask web framework on a local host.
#### TECH STACK: Selenium, gensim, tensorflow, sklearn, nltk, pandas, matplotlib, numpy, Flask, Jupyter lab, VSCode.

## 1. Web Scraping
* Data collection is done by scraping electronic products like Cameras and their accessories, headphones, mobile phones, laptops, and tablets from Amazon.
* Dynamic Scraping action implemented using Selenium and the retrieved data was written into CSV files
* TECH STACK: Selenium, BeautifulSoup

## 2. Sentiment classification LSTM model
* The classification model is trained on a large third-party dataset from Kaggle.
* The imported data is cleaned, wrangled, and undergoes basic feature engineering and analysis.
* Cleaned data is then transformed: NLTK tokenized data trains Word2Vec, generating the embedding matrix, while Keras tokenized data is used to train the model.
* The model achieved an accuracy, precision, recall, and F1 score of 0.89.
* TECH STACK: gensim, tensorflow, nltk, sklearn, pyspellchecker, re, numpy, pandas, matplotlib, Jupyter lab, VS Code. 
#### LSTM model classification report

![Screenshot (184)-min](https://github.com/arinsharma123/Cantilever-Web-Srape/assets/128144029/7efe49f1-2d3d-40b3-99e1-fe1d673c474d)

## 3. Data cleaning and wrangling (text preprocessing)
* This section focuses on NLP and text cleaning, primarily dealing with text data.
* The raw data undergoes extensive cleaning, including decontraction, removal of emojis, HTML tags, special characters, and punctuation.
* Multilingual reviews are detected and translated into English.
* Post-cleaning, reviews are classified as positive or critical using an LSTM model.
* TF-IDF vectorization and POS tagging are employed to extract key adjectives from reviews. 
* The fully preprocessed data, containing accurately classified information, is then exported.
* TECH STACK: tensorflow, nltk, sklearn, BeautifulSoup, langdetect, googletrans, re, seaborn, numpy, pandas, matplotlib.
```python
word2vec_items_model = Word2Vec(all_tokens, vector_size=50, window=5)
````

## 4. Data Viz and EDA
* Feature creation and basic Viz techniques to explore the data.
* TECH STACK: Seaborn, matplotlib, pandas.
  
## 5. Flask Web Application
* Basic Flask, HTML, and CSS used to design a web application hosted on localhost.
* Models imported to implement the Seach Engine Functionality (Word2Vec based).
* Query fetched from the home.html is processed to give the top 10 matching products and the results are displayed on results.html.
* Data Viz are available on the page data_viz.html.
* TECH STACK: Flask, sklearn, tensorflow, nltk, gensim, pandas, HTML5, CSS.
```python
@app.route('/')
def home():
    return render_template('home.html')

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
```
### Result page for a query
![Screenshot (183)](https://github.com/arinsharma123/Cantilever-Web-Srape/assets/128144029/ce24a7dd-421f-414d-a498-0a9f9a3dcb28)
![Scroll](https://github.com/arinsharma123/Cantilever-Web-Srape/assets/128144029/7ec9a49b-6579-4ef2-8979-89a8e34895fa)

