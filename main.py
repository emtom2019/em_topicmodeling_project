# Python interactive Code module
#%%

# Importing all modules needed 
from time import time
from datetime import datetime

import numpy as np
import pandas as pd
import re, nltk, gensim, spacy
import scispacy

#sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from pprint import pprint

#plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

#Personal modules
import skl_LDA_model_tuning as LT
import data_nl_processing as NLP

print('Modules Loaded')
#%%
# Loading files
print("Loading dataset...")
t0 = time()
df = pd.read_csv('data/external/data_methods_split.csv')
data_abstract = df['title_abstract'].tolist()
data_methods = df['methods'].tolist()
print("done in %0.3fs." % (time() - t0))

#%%
#Running NLP processing for LDA model input
nlp_methods = NLP.NlpForLdaInput(
                    data=data_methods, 
                    stopwords='english', 
                    spacy_lib='en_core_sci_md',
                    ngram_range=(1,1), 
                    min_df=10, 
                    max_df=0.25
                    )
nlp_abstract = NLP.NlpForLdaInput(
                    data=data_abstract, 
                    stopwords='english', 
                    spacy_lib='en_core_sci_md',
                    ngram_range=(1,1), 
                    min_df=10, 
                    max_df=0.25
                    )
nlp_methods.start()
nlp_abstract.start()
print("NLP processing done")

#%%
# Model fitting for abstract
abstract_tuner = LT.LdaTuning(
                data=nlp_abstract.lda_input(), 
                search_params={
                    'n_components':[5,10,15,20,25,30,35,40,45,50], 
                    'learning_decay':[.5,.7,.9], 
                    'learning_method': ['online']
                    }
                    )
abstract_tuner.start()
now = datetime.now().strftime("_%m%d%Y%H%M")
abstract_tuner.save_tuner('models/abstract_lda_tuner'+now)
print("Tuner saved to: " + 'models/abstract_lda_tuner'+now)
print("Abstract tuner results:")
abstract_tuner.compare_models(graph=True)
abstract_tuner.print_best_model()

#%%
# Model fitting for methods
methods_tuner = LT.LdaTuning(
                data=nlp_methods.lda_input(), 
                search_params={
                    'n_components':[5,10,15,20,25,30,35,40,45,50], 
                    'learning_decay':[.5,.7,.9], 
                    'learning_method': ['online']
                    }
                    )
methods_tuner.start()
now = datetime.now().strftime("_%m%d%Y%H%M")
methods_tuner.save_tuner('models/methods_lda_tuner'+now)
print("Tuner saved to: " + 'models/methods_lda_tuner'+now)
print("Methods tuner results:")
methods_tuner.compare_models(graph=True)
methods_tuner.print_best_model()

#%%
# Top 20 words for topics in abstract using best model
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ", ".join([feature_names[i]
                                for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
        print()

abstract_feature_names = nlp_abstract.get_feature_names()
print_top_words(abstract_tuner.best_model, abstract_feature_names, 20)   

#%%
# Top 20 words for topics in methods using best model
methods_feature_names = nlp_methods.get_feature_names()
print_top_words(methods_tuner.best_model, methods_feature_names, 20)   


# %%
