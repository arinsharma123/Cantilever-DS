# Amazon Web Scrape and Search Engine model
### Overview:
This project implements a full data processing and search engine pipeline, starting with web scraping for data collection. The data undergoes wrangling and thorough cleaning. An LSTM model, trained on a third-party dataset with 89% accuracy, classifies sentiment in the scraped reviews. Word2Vec embeddings are used to train a semantic search model, which provides the top 10 matching products for subjective and informal user queries. The project also includes a brief analysis of combined textual and numerical data using data visualization to understand user behavior for specific product types. The models are deployed using the Flask web framework on a local host.
#### TECH STACK: Selenium, gensim, tensorflow, sklearn, nltk, pandas, matplotlib, numpy, Flask.

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

## 3. Data cleaning and wrangling (text preprocessing)
* This section focuses on NLP and text cleaning, primarily dealing with text data.
* The raw data undergoes extensive cleaning, including decontraction, removal of emojis, HTML tags, special characters, and punctuation.
* Multilingual reviews are detected and translated into English.
* Post-cleaning, reviews are classified as positive or critical using an LSTM model.
* TF-IDF vectorization and POS tagging are employed to extract key adjectives from reviews. 
* The fully preprocessed data, containing accurately classified information, is then exported.
* TECH STACK: tensorflow, nltk, sklearn, BeautifulSoup, langdetect, googletrans, re, seaborn, numpy, pandas, matplotlib.

## 4. Data Viz and EDA
* Feature creation and basic Viz techniques to explore the data.
* TECH STACK: Seaborn, matplotlib, pandas.
  
## 5. Flask Web Application
* Basic Flask, HTML, and CSS used to design a web application hosted on localhost.
* Models imported to implement the Seach Engine Functionality (Word2Vec based).
* Query fetched from the home.html is processed to give the top 10 matching products and the results are displayed on results.html.
* Data Viz are available on the page data_viz.html.
* TECH STACK: Flask, sklearn, tensorflow, nltk, gensim, pandas, HTML5, CSS.
