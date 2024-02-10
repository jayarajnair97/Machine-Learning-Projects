
#Building a news classification system involves several steps, including web scraping, data preprocessing, and model training. Below is a high-level outline of the 
#steps you can follow using Natural Language Processing (NLP) techniques:

# Need for project
# For a number of reasons, a news classifier machine learning project may be quite beneficial.
#Content curation: It can be difficult for people to locate reliable and pertinent information online due to the deluge of news stories accessible. 
#To ensure that readers receive material that is relevant to their interests, a news classifier can assist in curating articles based on particular subjects.
#Personalization: A news classifier can provide readers with news suggestions that are tailored to their interests and reading patterns by examining their preferences and behavior.
#Fake News Detection: A news classifier can assist in identifying and flagging untrustworthy sources or articles, encouraging media literacy and combatting disinformation in light of the spread of fake news and misinformation.
#Topic Trend Analysis: A news classifier can assist with this by grouping news stories into several subjects or themes.

from selenium import webdriver
from bs4 import BeautifulSoup

# Define the URL of the BBC news website
url = "https://www.bbc.com/news"

# Initialize a Selenium webdriver 
driver = webdriver.Chrome() 

# Open the URL in the browser
driver.get(url)

# Get the page source after waiting for some time for dynamic content to load
driver.implicitly_wait(10)
html = driver.page_source

# Close the webdriver
driver.quit()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html, "html.parser")

# Find all news headlines and print them
headlines = soup.find_all("h3", class_="gs-c-promo-heading__title")
for headline in headlines:
    print(headline.text.strip())

# Import NLTK (Natural Language Toolkit)
import nltk
# Download NLTK resources (optional)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Import spaCy
import spacy

# Import gensim
import gensim

# Import TextBlob
from textblob import TextBlob

# Import scikit-learn
import sklearn

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize the WordNet lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean and preprocess the text
def preprocess_text(text):
    # Remove HTML tags
    clean_text = re.sub('<[^>]*>', '', text)
    # Remove non-alphanumeric characters and convert to lowercase
    clean_text = re.sub('[^a-zA-Z0-9]', ' ', clean_text).lower()
    # Tokenize the text
    tokens = word_tokenize(clean_text)
    # Remove stop words and single characters
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join the tokens back into a single string
    clean_text = ' '.join(lemmatized_tokens)
    return clean_text

# Clean and preprocess each headline
cleaned_headlines = [preprocess_text(headline.text) for headline in headlines]

# Print the cleaned headlines
print("Cleaned headlines:")
for headline in cleaned_headlines:
    print(headline)

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  

# Fit and transform the cleaned headlines using TF-IDF vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_headlines)

# Convert the TF-IDF matrix to an array
tfidf_array = tfidf_matrix.toarray()

# Print the shape of the TF-IDF array
print("TF-IDF array shape:", tfidf_array.shape)

from sklearn.cluster import KMeans
import numpy as np

# Defining the number of clusters based on requirements
num_clusters = 4  # Adjust this based on the topics you want to identify

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(tfidf_array)

# Print the cluster labels
print("Cluster labels:")
print(cluster_labels)

# Print the top words for each cluster
print("\nTop words for each cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = np.array(tfidf_vectorizer.get_feature_names_out())  # Accessing vocabulary directly
for i in range(num_clusters):
    print("Cluster {}:".format(i))
    top_words = terms[order_centroids[i, :10]]  # Print top 10 words for each cluster
    print(top_words)

# Define topic labels based on manual inspection of articles in each cluster
topic_labels = {
    0: "Environment",
    1: "Business",
    2: "Politics",
    3: "Culture"
}

# Print the assigned topic labels for each cluster
print("Assigned topic labels for each cluster:")
for cluster_id, topic_label in topic_labels.items():
    print("Cluster {}: {}".format(cluster_id, topic_label))

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Assuming 'cluster_labels' contains the cluster labels assigned to each headline
# Split the data into features (TF-IDF array) and target (cluster labels)
X = tfidf_array
y = cluster_labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the classifier (Naive Bayes)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Convert cluster labels to one-hot encoded vectors
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(cluster_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_array, encoded_labels, test_size=0.2, random_state=42)

# Define the maximum sequence length
max_length = 100  # Adjust as needed based on the maximum number of words in a headline

# Pad sequences to ensure uniform length
X_train_pad = pad_sequences(X_train, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test, maxlen=max_length, padding='post')

# Defining the LSTM model
model = Sequential()
model.add(Embedding(input_dim=X_train_pad.shape[1], output_dim=128, input_length=max_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))  # 4 output units for 4 clusters/topics

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_data=(X_test_pad, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print("Accuracy:", accuracy)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluating the performance of the classification model on the testing set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# You may experiment with other topologies, fine-tune the model parameters if necessary, or carry out more thorough 
# hyperparameter tuning using methods like grid search or random search.


import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Define the classification model
classifier = MultinomialNB()  # Initialize the Naive Bayes classifier

# Function to preprocess text
def preprocess_text(text):
    # Write your preprocessing code here
    return text

# Function to predict the topic
def predict_topic(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Vectorize the text using the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()  # You should initialize the vectorizer with the same parameters used during training
    text_vectorized = tfidf_vectorizer.transform([processed_text])
    # Predict the topic
    topic_prediction = classifier.predict(text_vectorized)[0]
    return topic_prediction

# Streamlit UI
st.title('News Topic Classifier')

# Input text area for user input
user_input = st.text_area("Enter the news headline:", "")

# Button to trigger prediction
if st.button('Predict'):
    # Perform prediction
    topic_prediction = predict_topic(user_input)
    st.write('Predicted Topic:', topic_prediction)

