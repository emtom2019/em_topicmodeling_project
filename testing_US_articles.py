from time import time
from datetime import datetime
import os, sys

import numpy as np
from scipy.stats.mstats import gmean
import pandas as pd
import pickle
import gensim
import spacy
import scispacy

import model_utilities as mu
import model_figure_pipeline as mfp
import data_nl_processing_v2, optimize_mallet, compare_models

if __name__ == "__main__": # Code only runs if this file is run directly. This is for testing purposes

    if False:
        df_raw = pd.read_csv("C:/Users/porto/Downloads/ultrasound_search/citation_ultrasound.csv", header=1, index_col=0)
        df = df_raw.filter(items = ['AB', 'SO', 'TI','YR'])
        df['SO'] = df['SO'].str.split(r'\.|=').str[0]
        df = df.rename(index=str, columns={"AB": "abstract", "SO": "journal", "TI": "title", "YR":"year"})
        df = df.reset_index()
        

        # add column with title + abstract
        df['title_abstract'] = df[['title', 'abstract']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

        df = df.filter(items = ['title', 'abstract', 'title_abstract', 'journal', 'year'])
        df['year'].fillna(int(0), inplace=True)
        year_replace = [int(x[0:4]) for x in df_raw['DC'].astype(str).tolist()]
        df['year'] = np.where(df['year'] == 0, year_replace, df['year'])
        df = df.reset_index(drop=True)
        df.to_csv('data/processed/ultrasound.csv',index=False)

    if False:
        data_path = 'data/processed/ultrasound.csv'
        df = pd.read_csv(data_path)
        data = df['title_abstract'].tolist()

        with mu.Timing('Processing Data...'):
            nlp_params = dict(spacy_lib='en_core_sci_lg', max_df=.25, bigrams=True, trigrams=True, max_tok_len=30)
            nlp_data = data_nl_processing_v2.NlpForLdaInput(data, **nlp_params)
            nlp_data.start()
        topic_range = (5,50,5)
        add_info='run1s'
        model_seed = int(time()*100)-158000000000
        compare_models = compare_models.CompareModels(nlp_data, topics=topic_range, seed=model_seed )
        compare_models.start()
        compare_models.save('reports/us/t({}_{}_{}){}{}mod'.format(*topic_range, add_info, model_seed))
        compare_models.output_dataframe(save=True, path='reports/us/t({}_{}_{}){}{}coh.csv'.format(*topic_range, add_info, model_seed))
        compare_models.output_dataframe(save=True, path='reports/us/t({}_{}_{}){}{}time.csv'.format(*topic_range, add_info, model_seed), data_column="time")
        compare_models.output_parameters(save=True, path='reports/us/t({}_{}_{}){}{}para.txt'.format(*topic_range, add_info, model_seed))
        compare_models.graph_results(show=False, save=True, path='reports/us/t({}_{}_{}){}{}.png'.format(*topic_range, add_info, model_seed))

    if False:
        data_path = 'data/processed/ultrasound.csv'
        data_column = 'title_abstract'
        addendum = "_us"
        model_save_folder = 'reports/us/models/'
        figure_save_folder = 'reports/us/figures/'
        topic_num = 15
        model_params = [
                    {'alpha':5,'optimize_interval':0},
                    {'alpha':10,'optimize_interval':0},
                    {'alpha':25,'optimize_interval':0},
                    {'alpha':50,'optimize_interval':0},
                    {'alpha':5,'optimize_interval':200},
                    {'alpha':10,'optimize_interval':200},
                    {'alpha':25,'optimize_interval':200},
                    {'alpha':50,'optimize_interval':200}]
        mu.generate_mallet_models(data_path, data_column, model_save_folder, figure_save_folder, topic_num, model_params, 
                            file_name_append=addendum)

    if True:
        model_path = 'reports/us/models/mallet_t15a25o200_us'
        data_path = 'data/processed/ultrasound.csv'
        data_column = 'title_abstract'
        year_column = 'year'
        journal_column = 'journal'
        topic_names = {
            1:"T1:Cardiac", 2:"T2:Doppler", 3:"T3:Image Asses", 4:"T4:Pediatrics", 5:"T5:Residency", 6:"T6:Diagnost Acc", 
            7:"T7:Med Ed", 8:"T8:Frequency", 9:"T9:Measure/Pred", 10:"T10:POC/Costs", 11:"T11:diag Decision", 
            12:"T12:Cath/Proc", 13:"T13:Lung/Thorax", 14:"T14:Evidence/review", 15:"T15:Abdominal Pain"
        }

        with mu.Timing('Running pipline...'):
            mfp.run_pipeline(model_path, data_path, data_column, year_column, journal_column, year_start=1976, year_res=5, year_end=2020,
                    docs=None, topic_names=topic_names, main_path='reports/us/')