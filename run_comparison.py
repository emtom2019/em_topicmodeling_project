from time import time
from datetime import datetime

import numpy as np
import pandas as pd
import re, gensim, spacy
import scispacy
import pickle
import nltk
from nltk.corpus import stopwords


#sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV, KFold
from pprint import pprint

#plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

import data_nl_processing
from silent_print import SilentPrinting
from compare_models import CompareModels

def run_model_comparison(runs, data_path, data_column, topic_range, add_info, models, ngram_max=3):
    for run in range(runs):
        
        if ngram_max > 2:
            trigrams = True
            bigrams = True
        elif ngram_max > 1:
            trigrams = False
            bigrams = True
        else:
            trigrams = False
            bigrams = False

        print("Loading dataset for CompareModels testing...")
        t0 = time()
        df = pd.read_csv(data_path)
        data = df[data_column].tolist()
        print("done in %0.3fs." % (time() - t0))

        stop_words = set(stopwords.words('english'))
        stop_words.update(['elsevier', 'copyright'])
        spacy_library = 'en_core_sci_lg'
        nlp_data = data_nl_processing.NlpForLdaInput(data, spacy_lib=spacy_library, max_df=.25, bigrams=bigrams, trigrams=trigrams)
        nlp_data.start()

        model_seed = int(time()*100)-158000000000

        compare_models = CompareModels(nlp_data=nlp_data, topics=topic_range, seed=model_seed, coherence='c_v', **models)
        compare_models.start()

        now = datetime.now().strftime("%m%d%Y%H%M")
        print("All models done at: " + now)
        
        compare_models.save('models/t({}_{}_{}){}{}mod'.format(*topic_range, add_info, model_seed))
        compare_models.output_dataframe(save=True, path='reports/t({}_{}_{}){}{}coh.csv'.format(*topic_range, add_info, model_seed))
        compare_models.output_dataframe(save=True, path='reports/t({}_{}_{}){}{}time.csv'.format(*topic_range, add_info, model_seed), data_column="time")
        compare_models.output_parameters(save=True, path='reports/t({}_{}_{}){}{}para.txt'.format(*topic_range, add_info, model_seed))
        compare_models.graph_results(show=False, save=True, path='reports/figures/t({}_{}_{}){}{}.png'.format(*topic_range, add_info, model_seed))

if __name__ == "__main__":
    data_path_all = 'data/external/data_cleaned.csv'
    data_column = 'title_abstract'
    topic_range = (5, 100, 5)
    models_all = {'gensim_lda':True, 'mallet_lda':True, 'sklearn_lda':True}
    run_model_comparison(2, data_path_all, data_column, topic_range, "a3g", models_all)

    data_path_title = 'data/processed/data_methods_split.csv'
    data_column_title = 'title_abstract'
    topic_range_title = (5, 100, 5)
    models_title = {'gensim_lda':False, 'mallet_lda':True, 'sklearn_lda':False}
    run_model_comparison(10, data_path_title, data_column_title, topic_range_title, "t3g", models_title)

    data_path_methods = 'data/processed/data_methods_split.csv'
    data_column_methods = 'methods'
    topic_range_methods = (5, 100, 5)
    models_methods = {'gensim_lda':False, 'mallet_lda':True, 'sklearn_lda':False}
    run_model_comparison(10, data_path_methods, data_column_methods, topic_range_methods, "m3g", models_methods)


