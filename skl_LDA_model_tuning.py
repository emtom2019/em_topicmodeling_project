from time import time
from datetime import datetime

import numpy as np
import pandas as pd
import re, gensim, spacy
import scispacy
import pickle
import nltk
from nltk.corpus import stopwords

#from lemmatization_methods import nltk_lemmatizer

#sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV, KFold
from pprint import pprint

#plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt


# Class for tuning the LDA model from sklearn by running it with various sets of parameters
# Initialize class with the vectorized data and a dictionary of the parameters that you want to evaluate
# Run the model with the .start() method. The best model will be saved as .best_model and creates a dataframe of the model comparison results
# .print_best_model() method prints outs the results for the best model and the parameters
# .compare_models(table=True, graph=False, x_label=None) prints out a table of the results by default. 
# Graph=True prints out a graph of the models with mean scores as the y component and the x_label as the x component.
# If x_label is not entered, than the first entry in the search_params is assumed to be the x_component

class MyLDAWithPerplexityScorer(LatentDirichletAllocation):

    def score(self, X, y=None):

        # You can change the options passed to perplexity here
        score = super(MyLDAWithPerplexityScorer, self).perplexity(X, sub_sampling=False)

        # Since perplexity is lower for better, so we do negative
        return -1*score

class LdaTuning:
    def __init__(self, data, search_params={'n_components':[5,10,15,20,25,30,35,40,45,50], 'learning_decay':[.5,.7,.9], 'learning_method': ['online']}):
        self.data = data
        self.search_parems = search_params
        #self.lda = MyLDAWithPerplexityScorer()
        self.lda = LatentDirichletAllocation()
        self.model = GridSearchCV(self.lda, 
                                    param_grid=self.search_parems,
                                    cv=KFold(n_splits=5, shuffle=True, random_state=None), 
                                    n_jobs=-1
                                    )
        self.best_model = None
        self.results = None

    def start(self):
        print('Initiating model fitting (may take a while)...')
        t0 = time() 
        self.model.fit(self.data)
        # Saves best model to self.best_model
        self.best_model = self.model.best_estimator_
        # Makes a pandas dataframe of the results of the model fits
        self.results = pd.DataFrame(self.model.cv_results_)
        print("done in %0.3fs." % (time() - t0))

    def print_best_model(self):

        # Best model parameters
        print("Best Model's Parameters: ", self.model.best_params_)

        # Log Likelihood Score
        print("Best Log Likelihood Score: ", self.model.best_score_)

        # Perplexity
        print("Best Model's Perplexity: ", self.best_model.perplexity(self.data))

    def export_best_model(self):
        return self.best_model

    def save_best_model(self, file_path_name):   
        with open(file_path_name, 'wb') as file:
            pickle.dump(self.best_model, file)

    def save_tuner(self, file_path_name):
        with open(file_path_name, 'wb') as file:            
            pickle.dump(self, file)

    def compare_models(self, table=True, graph=False, x_label=None):
        # Print out comparisons of all of the models        
        if table == True:
            print(self.results[['mean_test_score', 'std_test_score', 'params']])

        # Graphs model score versus x paramater
        if graph == True:
            graphs = {}
            n_models = range(len(self.results['params']))
            # Extracts X values from the given params    
            if x_label == None:
                x_label = list(self.search_parems)[0]
            # Builds the graph labels as params minus the X parameter
            for params in self.results['params']:
                label = params.copy()
                label.pop(x_label)
                graphs[str(label)] = [[],[]]
            # Fills in the data for each graph
            for i in n_models:
                label = self.results.loc[i]['params'].copy()
                x_value = label.pop(x_label)
                y_value = self.results.loc[i]['mean_test_score']
                graphs[str(label)][0].append(x_value)
                graphs[str(label)][1].append(y_value)
            
            # Show graph
            plt.figure(figsize=(12, 8))
            for label in graphs:
                plt.plot(graphs[label][0], graphs[label][1], label=label)
            plt.title("Choosing Optimal LDA Model")
            plt.xlabel(x_label)
            plt.ylabel("Log Likelyhood Scores")
            plt.legend(title='Parameters', loc='best')
            plt.show()

if __name__ == "__main__": # Prevents the following code from running when importing module
    #import dataset
    print("Loading dataset...")
    t0 = time()
    #df = pd.read_csv('data/external/data_methods_split.csv')
    df = pd.read_csv('data/external/data_cleaned.csv')
    data = df["title_abstract"].tolist()
    print("done in %0.3fs." % (time() - t0))

    print("Preprocessing dataset...")
    print('Creating Tokens...')
    t0 = time()

    #process dataset:
    #Tokenize
    def sent_to_words(sentences):
        for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence),deacc=True)) #deacc=True removes punctuations
    data_words = list(sent_to_words(data[:]))
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
    stop_words = set(stopwords.words('english'))
    stop_words.update(['elsevier', 'copyright', 'rights', 'reserve', 'reserved', 'ed'])
    vectorizer = CountVectorizer(analyzer='word',               # features are 'word' vs 'char' or 'char_wb' (char_wb or only characters inside words)
                            min_df=10,                          # minimum number of word repeats
                            stop_words=stop_words,               # removes stop words (can also use max_df)
                            lowercase=True,                     # converts everything to lowercase
                            token_pattern='[a-zA-Z0-9]{2,}',    # char considered and max length
                            ngram_range=(1,1),                   # n-gram range (min, max)
                            #max_features=50000,                # Max number of features (words from highest freq to lowest)
                            max_df=0.5                        # Removes words that occur in more than n proportion of texts
                            )

    data_vectorized = vectorizer.fit_transform(data_lemmatized_spacy)
    print(vectorizer)
    #checking sparcity because it was in my tutorial (is this an important thing?)
    data_dense = data_vectorized.todense()
    print('Sparcity: ', ((data_dense > 0).sum()/data_dense.size)*100, '%')
    print("done in %0.3fs." % (time() - t0))

    custom_search_params = {'n_components':[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], 'learning_decay':[.7], 'learning_method': ['online']}
    lda_tuning = LdaTuning(data_vectorized, search_params=custom_search_params)
    lda_tuning.start()
    lda_tuning.print_best_model()
    now = datetime.now().strftime("_%m%d%Y%H%M")
    lda_tuning.save_tuner('models/lda_tuner'+now)

    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += ", ".join([feature_names[i]
                                for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()

    tf_feature_names = vectorizer.get_feature_names()
    print_top_words(lda_tuning.best_model, tf_feature_names, 20)   


    lda_tuning.compare_models(graph=True)
    print(lda_tuning.best_model.get_params())
    lda_tuning.print_best_model()
