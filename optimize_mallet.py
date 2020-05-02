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
from pprint import pprint

#plotting tools
import matplotlib.pyplot as plt

import data_nl_processing 
from model_utilities import SilentPrinting

class CompareMalletModels:
    def __init__(self, nlp_data, topics=[40], alpha=[50], opt_int=[0], iterations=[1000],
                repeats=1, topn=20, coherence='c_v', mallet_path='C:\\mallet\\bin\\mallet'):
        self.nlp_data = nlp_data
        self.repeats = repeats
        self.mallet_path = mallet_path
        self.models = {}
        for t in topics:
            for a in alpha:
                for o in opt_int:
                    for i in iterations:
                        self.models['t{}a{}o{}i{}'.format(t, a, o, i)] = []
        self.coherence = coherence
        self.topn = topn
        self.testing_para = {
            'topics':topics,
            'alpha':alpha,
            'opt_int':opt_int,
            'iterations':iterations
        }
        self.parameters = {
            'mallet_path':self.mallet_path,
            'workers':5,
        }

    def start(self, verbose=True):
        print("Initiating model building...")
        t0 = time()
        with SilentPrinting(verbose):
            self.run_models_()
        print("All models done in {:0.3f}s".format(time()-t0))

    def run_models_(self):
        for run in range(self.repeats):
            t1 = time()
            seed = int(t1*100)-158000000000
            print("Initiating run {}/{}".format(run+1, self.repeats))
            parameters = {
                'mallet_path':self.parameters['mallet_path'],
                'workers':self.parameters['workers']
            }
            for t in self.testing_para['topics']:
                for a in self.testing_para['alpha']:
                    for o in self.testing_para['opt_int']:
                        for i in self.testing_para['iterations']:
                            parameters['num_topics'] = t
                            parameters['alpha'] = a
                            parameters['optimize_interval'] = o
                            parameters['iterations'] = i
                            parameters['random_seed'] = seed
                            model_name = 't{}a{}o{}i{}'.format(t, a, o, i)
                            print("Running model {} run {}...".format(model_name, run+1))
                            self.models[model_name].append(self.mallet_model_(parameters))

            print("Run {}/{} done in {:0.3f}s.".format(run+1, self.repeats, time() - t1))


    def mallet_model_(self, parameters):
        tm = time()
        print("Building Mallet model...")     
        model = gensim.models.wrappers.LdaMallet(corpus=self.nlp_data.gensim_lda_input(),
                                                    id2word=self.nlp_data.get_id2word(),
                                                    **parameters)

        model_time =  (time() - tm)                                       
        print("Done in %0.3fs." % model_time)
        tc = time()
        print("Running Coherence model for Mallet...")
        model_topics_list = self.gensim_topic_words_(model.show_topics(formatted=False, num_words=self.topn, num_topics=-1))
        coh_model = gensim.models.CoherenceModel(topics=model_topics_list, texts=self.nlp_data.get_token_text(),
                                                        dictionary=self.nlp_data.get_id2word(), window_size=None,
                                                        coherence=self.coherence)
        model_coherence = coh_model.get_coherence()

        print("Done in %0.3fs." % (time() - tc))
        print("Mallet coherence for {} topics is: {:0.3f}".format(parameters['num_topics'], model_coherence))
        return {"coherence":model_coherence, "time":model_time, "topics":model_topics_list, "model":model, 'seed':parameters['random_seed']}

    def gensim_topic_words_(self, show_topics):
        show_topics.sort()
        topic_word_list = []
        for topic in show_topics:
            message = "Topic #%d: " % topic[0]
            new_list = list(word[0] for word in topic[1])
            message += ", ".join(new_list)
            topic_word_list.append(new_list)
            print(message)
        print()
        return topic_word_list

    def graph_results(self, show=True, save=False, path=None):
        graphs = self.output_dataframe(seed=False).to_dict('list')
        labels = list(graphs.keys())
        x_values = []
        for label in labels:
            x_values.append(graphs[label])

        # Show graph
        plt.figure(figsize=(12, 8))    
        plt.boxplot(x_values, labels=labels)

        plt.title("Mallet Model Comparison")
        plt.xlabel("Models: t:topics, a:alpha, o:optimize interval, i:iterations")
        plt.ylabel(self.coherence + " coherence")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save:
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close() # Closes and deletes graph to free up memory

    def output_dataframe(self, data_column="coherence", seed=True, save=False, path=None):
        dataframe_dict = {}
        if seed:
            seed_filled = False
            dataframe_dict['seed'] = []
        else:
            seed_filled = True
        for key in self.models:
            dataframe_dict[key] = []
            for model in self.models[key]:
                dataframe_dict[key].append(model[data_column])
                if not seed_filled:
                    dataframe_dict['seed'].append(model['seed'])
            seed_filled = True
        data = pd.DataFrame.from_dict(dataframe_dict)
        if save:
            data.to_csv(path, index=False)
        return data

    def save(self, file_path_name):
        with open(file_path_name, 'wb') as file:            
            pickle.dump(self, file)

def graph_results(CompareMallets, show=True, save=False, path=None): 
    # This is for graphing with better x label placement of the file models/mallet_model_comparisons_1
    # Newer mallet comparisons have the fix
    graphs = CompareMallets.output_dataframe(seed=False).to_dict('list')
    labels = list(graphs.keys())
    x_values = []
    for label in labels:
        x_values.append(graphs[label])

    # Show graph
    plt.figure(figsize=(12, 8))    
    plt.boxplot(x_values, labels=labels)

    plt.title("Mallet Model Comparison")
    plt.xlabel("Models: t:topics, a:alpha, o:optimize interval, i:iterations")
    plt.ylabel(CompareMallets.coherence + " coherence")
    plt.xticks(rotation=45, ha='right')
    if save:
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close() # Closes and deletes graph to free up memory

if __name__ == "__main__":

    build_model = True
    if build_model:
        data_path = 'data/external/data_cleaned.csv'
        data_column = 'title_abstract'
        print("Loading dataset for CompareModels testing...")
        t0 = time()
        df = pd.read_csv(data_path)
        data = df[data_column].tolist()
        print("done in %0.3fs." % (time() - t0))

        spacy_library = 'en_core_sci_lg'
        nlp_data = data_nl_processing.NlpForLdaInput(data, spacy_lib=spacy_library, max_df=.25, bigrams=True, trigrams=True)
        nlp_data.start()

        models = CompareMalletModels(nlp_data, topics=[40], alpha=[5,10,25,50], opt_int=[0, 100, 200], iterations=[1000], repeats=20)
        models.start()
        models.save('models/mallet_model_comparisons_14')
        models.output_dataframe(save=True, path='reports/mallet_model_comparisons_14.csv')
        models.graph_results(show=False, save=True, path='reports/figures/mallet_model_comparisons_14.png')
        models.graph_results()

    load_model = False
    if load_model:
        with open('models/mallet_model_comparisons_1', 'rb') as model:
            mallet_models = pickle.load(model)
        graph_results(mallet_models)
