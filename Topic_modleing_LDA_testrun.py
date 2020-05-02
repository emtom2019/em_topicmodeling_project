from time import time

import numpy as np
import pandas as pd
import re, nltk, gensim, spacy
import scispacy

#from lemmatization_methods import nltk_lemmatizer

#sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from pprint import pprint

#plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

#import dataset
print("Loading dataset...")
t0 = time()
df = pd.read_csv('data/external/data_cleaned.csv')
data = df.title_abstract.values.tolist()
print("done in %0.3fs." % (time() - t0))

print("Preprocessing dataset...")
print('Creating Tokens...')
t0 = time()

#process dataset:
#Tokenize
def sent_to_words(sentences):
      for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence),deacc=True)) #deacc=True removes punctuations
data_words = list(sent_to_words(data[:100]))
print("done in %0.3fs." % (time() - t0))

#Lemmatization using spaCy package for lemmatization (simpler than NLTK)
#https://spacy.io/api/annotation
print('Lemmetization in progress using spaCy...')
t0 = time()
nlp = spacy.load('en_core_sci_md', disable=['parser','ner'])
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB','ADV']):
      output_text = []
      for abstract in texts:
            doc = nlp(" ".join(abstract))
            output_text.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc]))
      return output_text
            
data_lemmatized_spacy = lemmatization(data_words)
print("done in %0.3fs." % (time() - t0))
#print(data_lemmatized_spacy[:1])

# Vectorizing dataset for use with LDA algorithm
print('Vectorizing dataset...')
t0 = time()
vectorizer = CountVectorizer(analyzer='word',               # features are 'word' vs 'char' or 'char_wb' (char_wb or only characters inside words)
                        min_df=10,                          # minimum number of word repeats
                        stop_words='english',               # removes stop words (can also use max_df)
                        lowercase=True,                     # converts everything to lowercase
                        token_pattern='[a-zA-Z0-9]{2,}',    # char considered and max length
                        ngram_range=(1,1),                   # n-gram range (min, max)
                        #max_features=50000,                # Max number of features (words from highest freq to lowest)
                        max_df=0.7                        # Removes words that occur in more than n proportion of texts
                        )

data_vectorized = vectorizer.fit_transform(data_lemmatized_spacy)
print(vectorizer)
#checking sparcity because it was in my tutorial (is this an important thing?)
data_dense = data_vectorized.todense()
print('Sparcity: ', ((data_dense > 0).sum()/data_dense.size)*100, '%')
print("done in %0.3fs." % (time() - t0))

# Building the LDA model
print('Building LDA model...')
t0 = time()
lda_model = LatentDirichletAllocation(n_components = 30,              # Number of topics
                                    max_iter=10,                  # Max learning iterations
                                    learning_method='online',     # Learning method batch versus online
                                    random_state=100,             # Random state seed
                                    batch_size=128,               # Batch size for online learning
                                    evaluate_every=-1,            # Calculates perplexity every n iterations (it is off)
                                    n_jobs=-1,                    # Number of CPUs used (-1 = all)
                                    )

lda_output = lda_model.fit_transform(data_vectorized)
print(lda_model) # Prints model attributes
print("done in %0.3fs." % (time() - t0))

# Printing top 20 words for each topic
n_top_words = 20

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ", ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

tf_feature_names = vectorizer.get_feature_names()
print_top_words(lda_model, tf_feature_names, n_top_words)    

# Model Performance
# Log Likelyhood: Higher is better
print("Log Likelyhood: ", lda_model.score(data_vectorized))

# Perplexity: Lower is better. Perplexity = exp(-1.*log-likelihood per word)
print('Perplexity: ', lda_model.perplexity(data_vectorized))

# See model parameters
print(lda_model.get_params())

#Do not use NLTK for lemmetization, it is very slow and is not very good or my code is just bad
#print('Lemmetization in progress using NLTK...')
#t0 = time()
#data_lemmatized_nltk = nltk_lemmatizer(data_words)
#print("done in %0.3fs." % (time() - t0))
#print(data_lemmatized_nltk[:1])

#spacy_text_file = open('spacy_lem_sample.txt', 'w')
#spacy_text_file.write(data_lemmatized_spacy[0])
#spacy_text_file.close()

#nltk_text_file = open('nltk_lem_sample.txt', 'w')
#nltk_text_file.write(data_lemmatized_nltk[0])
#nltk_text_file.close()


'''
n_samples = 2000
n_features = 1000
n_components = 20
n_top_words = 20

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'), return_X_y=True)
data_samples = data[:n_samples]
print("done in %0.3fs." % (time() - t0))

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print()

# Fit the NMF model
print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

# Fit the NMF model
print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
      "tf-idf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
'''