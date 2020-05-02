
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

class CompareModels:
    def __init__(self, nlp_data, topics=(5,50,5), seed=None, gensim_lda=True, mallet_lda=True, sklearn_lda=True, 
                    topn=20, coherence='c_v', mallet_path='C:\\mallet\\bin\\mallet'):
        self.nlp_data = nlp_data
        self.gensim_lda = gensim_lda
        self.mallet_lda = mallet_lda
        self.sklearn_lda = sklearn_lda
        self.topics_range = range(topics[0], topics[1]+topics[2], topics[2])
        self.seed = seed
        self.mallet_path = mallet_path
        self.models = {}
        self.coherence = coherence
        self.topn = topn
        if self.seed is None:
            mallet_seed = 0
        else:
            mallet_seed = self.seed
        self.parameters = {
            'gensim':{
                'decay':0.5,
                'alpha':'asymmetric',
                'eta':None,
                'workers':None,
                'random_state':self.seed,
                'chunksize':100,
                'passes':10,
                'per_word_topics':True
            },
            'mallet':{
                'mallet_path':self.mallet_path,
                'workers':5,
                'random_seed':mallet_seed,
                'alpha':50,
                'iterations':1000,
                'optimize_interval':0
            },
            'sklearn':{
                'max_iter':10,                  # Max learning iterations
                'learning_method':'online',     # Learning method batch versus online
                'random_state':self.seed,       # Random state seed
                'batch_size':128,               # Batch size for online learning
                'evaluate_every':-1,            # Calculates perplexity every n iterations (it is off)
                'n_jobs':-1,                    # Number of CPUs used (-1 = all)
                'learning_decay':0.7,
                'learning_offset':10,
                'doc_topic_prior':None,         # alpha that defaults to 1 / n_components  
                'topic_word_prior':None         # eta that defaults to 1 / n_components              
            }
        }

    def start(self, verbose=True):

        print("Initiating model building...")
        t0 = time()
        if self.gensim_lda:
            self.models['Gensim'] = {}
        if self.mallet_lda:
            self.models['Mallet'] = {}
        if self.sklearn_lda:
            self.models['Sklearn'] = {}
        with SilentPrinting(verbose):
            self.run_models_(self.topics_range)
        print("All models done in {:0.3f}s".format(time()-t0))

    def run_models_(self, topics):
        for topic_num in topics:
            t1 = time()
            print("Building models with {} topics...".format(topic_num))
            
            if self.gensim_lda:
                self.models['Gensim'][topic_num] = self.gensim_model_(topic_num)

            if self.mallet_lda:
                self.models['Mallet'][topic_num] = self.mallet_model_(topic_num)
            
            if self.sklearn_lda:
                self.models['Sklearn'][topic_num] = self.sklearn_model_(topic_num)

            print("{}-topic models done in {:0.3f}s.".format(topic_num, time() - t1))



    def gensim_model_(self, topics):
        tm = time()

        print("Building Gensim model...")
        model = gensim.models.LdaMulticore(corpus=self.nlp_data.gensim_lda_input(),
                                                id2word=self.nlp_data.get_id2word(),
                                                num_topics=topics,
                                                **self.parameters['gensim'])
        model_time =  (time() - tm)                                       
        print("Done in %0.3fs." % model_time)
        tc = time()
        print("Running Coherence model for Gensim...")
        model_topics_list = self.gensim_topic_words_(model.show_topics(formatted=False, num_words=self.topn, num_topics=-1))
        coh_model = gensim.models.CoherenceModel(topics=model_topics_list, texts=self.nlp_data.get_token_text(),
                                                        dictionary=self.nlp_data.get_id2word(), window_size=None,
                                                        coherence=self.coherence)
        model_coherence = coh_model.get_coherence()

        print("Done in %0.3fs." % (time() - tc))
        print("Gensim coherence for {} topics is: {:0.3f}".format(topics, model_coherence))
        return {"coherence":model_coherence, "time":model_time, "topics":model_topics_list, "model":model}

    def mallet_model_(self, topics):
        tm = time()
        print("Building Mallet model...")     
        model = gensim.models.wrappers.LdaMallet(corpus=self.nlp_data.gensim_lda_input(),
                                                    id2word=self.nlp_data.get_id2word(),
                                                    num_topics=topics,
                                                    **self.parameters['mallet'])

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
        print("Mallet coherence for {} topics is: {:0.3f}".format(topics, model_coherence))
        return {"coherence":model_coherence, "time":model_time, "topics":model_topics_list, "model":model}

    def sklearn_model_(self, topics):
        tm = time()

        print("Building Sklearn model...")

        model = LatentDirichletAllocation(n_components = topics,      # Number of topics
                                        **self.parameters['sklearn']
                                        )
        model.fit(self.nlp_data.sklearn_lda_input())

        model_time =  (time() - tm)                                       
        print("Done in %0.3fs." % model_time)
        tc = time()
        print("Running Coherence model for Sklearn...")
        model_topics_list = self.sklearn_topic_words_(model)
        coh_model = gensim.models.CoherenceModel(topics=model_topics_list, texts=self.nlp_data.get_token_text(),
                                                        dictionary=self.nlp_data.get_id2word(), window_size=None,
                                                        coherence=self.coherence)
        model_coherence = coh_model.get_coherence()

        print("Done in %0.3fs." % (time() - tc))
        print("Sklearn coherence for {} topics is: {:0.3f}".format(topics, model_coherence))
        return {"coherence":model_coherence, "time":model_time, "topics":model_topics_list, "model":model}

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

    def sklearn_topic_words_(self, model):
        topic_word_list = []
        feature_names = self.nlp_data.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            new_list = list(feature_names[i]
                                for i in topic.argsort()[:-self.topn - 1:-1])
            message += ", ".join(new_list)
            topic_word_list.append(new_list)
            print(message)
        print()
        return topic_word_list
    def graph_results(self, show=True, save=False, path=None):
        graphs = {}
        # Bulding graph x,y lists
        for key in self.models:
            graphs[key] = ([],[])
            for topic_num in self.models[key]:
                graphs[key][0].append(topic_num)
                graphs[key][1].append(self.models[key][topic_num]['coherence'])
        
        # Show graph
        plt.figure(figsize=(12, 8))
        for key in graphs:
            plt.plot(graphs[key][0], graphs[key][1], label=key)
        plt.title("Choosing Optimal LDA Model")
        plt.xlabel("Number of topics")
        plt.ylabel(self.coherence + " coherence")
        plt.legend(title='Models', loc='best')
        if save:
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close() # Closes and deletes graph to free up memory

    def output_dataframe(self, data_column="coherence", save=False, path=None):
        dataframe_dict = {}
        topics = None
        for key in self.models:
            data_list = []
            if topics is None:
                topics = list(self.models[key].keys())
                topics.sort()
                dataframe_dict["Number of Topics"] = topics
            for topic_num in topics:
                data_list.append(self.models[key][topic_num][data_column])
            dataframe_dict[key] = data_list
        data = pd.DataFrame.from_dict(dataframe_dict)
        if save:
            data.to_csv(path, index=False)
        return data

    def output_parameters(self, save=False, path=None):
        if save:
            with open(path, 'w') as file:
                file.write(str(self.parameters))
        return self.parameters

    def save(self, file_path_name):
        with open(file_path_name, 'wb') as file:            
            pickle.dump(self, file)

if __name__ == "__main__":
    print("Loading dataset for CompareModels testing...")
    t0 = time()
    df = pd.read_csv('data/processed/data_methods_split.csv')
    data = df["title_abstract"].tolist()
    print("done in %0.3fs." % (time() - t0))

    stop_words = set(stopwords.words('english'))
    stop_words.update(['elsevier', 'copyright', 'rights', 'reserve', 'reserved', 'ed'])
    nlp_data = data_nl_processing.NlpForLdaInput(data)
    nlp_data.start()

    compare_models = CompareModels(nlp_data=nlp_data, topics=(20,30,5), seed=133, coherence='c_v')
    compare_models.start()
    compare_models.output_dataframe(save=True, path='reports/test_file.csv')
    compare_models.output_dataframe(save=True, path='reports/test_file2.csv', data_column="time")
    print(compare_models.output_parameters(save=True, path='reports/test_file3.txt'))
    compare_models.graph_results()