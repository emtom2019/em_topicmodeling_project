from time import time
from datetime import datetime
import os, sys

import numpy as np
import pandas as pd
import gensim, spacy, scispacy
import pickle
import glob
import random
import model_figure_pipeline as mfp

import data_nl_processing, data_nl_processing_v2
import model_utilities as mu

def coherence_set(model, coherence, window=None):
    model_topics_list = model.gensim_topic_words_(model.model.show_topics(formatted=False, num_words=20, num_topics=-1))
    coh_model = gensim.models.CoherenceModel(topics=model_topics_list, texts=model.nlp_data.get_token_text(),
                                                    dictionary=model.nlp_data.get_id2word(), window_size=window,
                                                    coherence=coherence)
    model_coherence = coh_model.get_coherence()

    return model_coherence

if __name__ == "__main__": # Code only runs if this file is run directly.
    if False: # Loading data and model
        t =  time()                                      
        print("Loading Data...")
        with open('models/main_mallet_t40a25o200', 'rb') as model:
            mallet_model = pickle.load(model)
        
        data_path = 'data/external/data_cleaned.csv'
        data_column = 'title_abstract'
        df = pd.read_csv(data_path)
        raw_text = df[data_column].tolist()
        year_list = df['year'].tolist()
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # docs per 1 year
        t =  time()                                      
        print("Running doc counts per time ...")
        df1, df2 = mu.doc_topics_per_time(mallet_model.model, mallet_model.nlp_data, year_list=year_list, year_res=1)
        df1.to_csv('reports/main_model/doc_n_per1_year.csv', index=False)
        df2.to_csv('reports/main_model/doc_w_per1_year.csv', index=False)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Total docs per topic
        t =  time()                                      
        print("Running total Doc counts per topic  ...")
        df3, df4 = mu.docs_per_topic(mallet_model.model, mallet_model.nlp_data)
        df3.to_csv('reports/main_model/doc_n_per_topic.csv', index=False)
        df4.to_csv('reports/main_model/doc_w_per_topic.csv', index=False)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Docs by dominant topic
        t =  time()                                      
        print("Running docs by dominant topic ...")
        topic_df = mu.dominant_doc_topic_df(mallet_model.model, mallet_model.nlp_data)    
        topic_df.to_csv('reports/main_model/docs_dom_topic.csv')
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

        t =  time()                                      
        print("Running best doc per topic  ...")
        best_doc_df = mu.best_doc_for_topic(topic_df)
        best_doc_df.to_csv('reports/main_model/best_doc_per_topic.csv')
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

        t =  time()                                      
        print("Running best doc per topic with raw test ...")
        doc_list = best_doc_df["Best Document"]
        new_column = []
        for doc in doc_list:
            new_column.append(raw_text[int(doc-1)])
        best_doc_raw_df = best_doc_df.copy()
        best_doc_raw_df["Raw Text"] = pd.Series(new_column).values
        best_doc_raw_df.to_csv('reports/main_model/best_doc_per_topic_with_raw.csv')
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Document token histogram
        t =  time()                                      
        print("Creating doc token counts ...")
        mu.plot_doc_token_counts(topic_df,fig_save_path='reports/main_model/doc_token_counts.png', show=False)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Reloading dataframes
        t =  time()                                      
        print("Reloading df data ...")
        data_path2 = 'reports/main_model/best_doc_per_topic_with_raw.csv'
        data_column2 = 'Raw Text'
        df_raw = pd.read_csv(data_path2)
        new_column = df_raw[data_column2].tolist()
        df1_data_path = 'reports/main_model/doc_n_per1_year.csv'
        df1 = pd.read_csv(df1_data_path)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Create sample colored paragraph
        t =  time()                                      
        print("Creating sample colored paragraph ...")
        mu.color_doc_topics(mallet_model.model, raw_text[0], mallet_model.nlp_data, topics=4, line_word_length=12,
                        fig_save_path='reports/main_model/sample_colordoctopics.png', show=False, custom_titles=mu.MAIN_TOPICS)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Creating colored docs of dominant topics
        t =  time()                                      
        print("Creating best colored paragraph per topic...")
        for i, text in enumerate(new_column):
            mu.color_doc_topics(mallet_model.model, text, mallet_model.nlp_data, topics=4, line_word_length=12,
                        fig_save_path='reports/main_model/best_colored_paragraphs/colordoctopic_{}.png'.format(i+1), show=False,
                        custom_titles=mu.MAIN_TOPICS)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)
    
    if False: # Creating colored docs of dominant topics
        t =  time()                                      
        print("Creating best colored paragraph per topic...")
        for i, text in enumerate(new_column):
            if i == 26:
                mu.color_doc_topics(mallet_model.model, text, mallet_model.nlp_data, topics=4, line_word_length=12,
                        fig_save_path='reports/main_model/best_colored_paragraphs/colordoctopic_{}.png'.format(i+1), show=False,
                        custom_titles=mu.MAIN_TOPICS)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Graph of number of docs per topic per year
        t =  time()                                      
        print("Creating graph of number of docs per topic per year...")
        x_val = list(range(1980,2020))
        mu.plot_doc_topics_per_time(df1, 40, 8, ylabel='Proportion of Documents', xlabel='Years', fig_save_path='reports/main_model/plot_n_abs_per1y.png', 
                                    x_val=x_val, hide_x_val=False, xtick_space=5, custom_titles=mu.MAIN_TOPICS, relative_val=True,
                                    df_data2=df1, relative_val2=False, ylabel2="Absolute Count of Documents", show=False)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)
    
    if False: # Graph of number of docs per topic per year
        t =  time()                                      
        print("Creating graph of number of docs per topic per year...")
        columns = list(df1.columns)[1:]
        column_totals = df1.loc[:,columns[0]:].sum(axis=0)
        column_totals_list = list(column_totals)
        topics_list = df1["Topic"]
        years = list(range(1980,2020))
        mu.graph(years, column_totals_list, title="Total Abstracts by Year", x_label="Year", y_label="Number of Abstracts", 
            fig_save_path='reports/main_model/total_docs_per_year.png')
        total_docs = 0
        for total in column_totals_list:
            total_docs += total
        print("Total docs: {}".format(total_docs))
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Graph of number of docs for every 5 years
        t =  time()                                      
        print("Running doc counts per time ...")
        df5, df6 = mu.doc_topics_per_time(mallet_model.model, mallet_model.nlp_data, year_list=year_list, year_res=5)
        df5.to_csv('reports/main_model/doc_n_per5_year.csv', index=False)
        df6.to_csv('reports/main_model/doc_w_per5_year.csv', index=False)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)    

    if False: # Reloading dataframes
        t =  time()                                      
        print("Reloading df data ...")
        df5_data_path = 'reports/main_model/doc_n_per5_year.csv'
        df5 = pd.read_csv(df5_data_path)
        df6_data_path = 'reports/main_model/doc_w_per5_year.csv'
        df6 = pd.read_csv(df6_data_path)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)
    
    if False: # Total abstracts for every 5 years
        t =  time()                                      
        print("Creating graph of number of docs per year...")
        columns = list(df5.columns)[1:]
        column_totals = df5.loc[:,columns[0]:].sum(axis=0)
        column_totals_list = list(column_totals)
        topics_list = df5["Topic"]
        years = list(range(1980,2020,5))
        mu.graph(columns, column_totals_list, title="Total Abstracts by Year", x_label="Years", y_label="Number of Abstracts", 
            fig_save_path='reports/main_model/total_docs_per_5year.png')
        total_docs = 0
        for total in column_totals_list:
            total_docs += total
        print("Total docs: {}".format(total_docs))
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Number of docs per topic per 5 years
        t =  time()                                      
        print("Creating graph of number of docs per topic per year...")
        x_val = list(range(1980,2020, 5))
        mu.plot_doc_topics_per_time(df5, 40, 8, ylabel='Proportion of Documents', xlabel='Years', fig_save_path='reports/main_model/plot_n_abs_per5y.png', 
                                    x_val=x_val, hide_x_val=False, xtick_space=10, xmintick_space=5, custom_titles=mu.MAIN_TOPICS_TRUNC, relative_val=True,
                                    df_data2=df5, relative_val2=False, ylabel2="Absolute Count of Documents", show=False)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Number of docs per topic per 5 years with weight
        t =  time()                                      
        print("Creating graph of document weight per topic per year...")
        x_val = list(range(1980,2020, 5))
        mu.plot_doc_topics_per_time(df5, 40, 8, ylabel='Proportion of Documents', xlabel='Years', fig_save_path='reports/main_model/plot_n_w_per5y.png', 
                                    x_val=x_val, hide_x_val=False, xtick_space=10, xmintick_space=5, custom_titles=mu.MAIN_TOPICS, relative_val=True,
                                    df_data2=df6, relative_val2=True, ylabel2=None, show=False)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Create several colored paragraph samples
        t =  time()                                      
        print("Creating sample colored paragraph ...")
        for i in range(10):
            index = random.randrange(20000)
            mu.color_doc_topics(mallet_model.model, raw_text[index], mallet_model.nlp_data, topics=3, line_word_length=12, incl_perc=True,
                        fig_save_path='reports/main_model/sample_color_docs/doc_{}.png'.format(index), show=False, custom_titles=mu.MAIN_TOPICS)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)
    
    if False:
        t =  time()                                      
        print("Creating t-SNE doc Cluster ...")
        seed = 2019
        mu.plot_tsne_doc_cluster(mallet_model.model, mallet_model.nlp_data, marker_size=1, min_tw=None, seed=seed, show_topics=True, 
            show=False, custom_titles=mu.MAIN_TOPICS_TRUNC, fig_save_path='reports/main_model/tsne_doc_cluster_s{}.png'.format(2019))
        comp_time =  (time() - t)   
        print("Done in %0.3fs." % comp_time)
    
    if False:
        t =  time()                                      
        print("Creating topic word clouds ...")
        mu.creat_multi_wordclouds(40, 8, mallet_model.model, mallet_model.nlp_data, custom_titles=mu.MAIN_TOPICS_TRUNC, show=False,
            fig_save_path='reports/main_model/topic_wordclouds.png', title_font=14)
        comp_time =  (time() - t)   
        print("Done in %0.3fs." % comp_time)

    if False: # Group data by journal
        t =  time()                                      
        print("Calculating df counts ...")
        df_dict, counts = mu.rows_per_df_grp(df, "journal")
        print(counts)
        comp_time =  (time() - t)   
        print("Done in %0.3fs." % comp_time)
        for journal in df_dict:
            y_list = df_dict[journal]['year'].tolist()
            y_list.sort()
            print("Journal: {}, First pub: {}".format(journal, y_list[0]))

    if False: # Document counts by year topic and journal
        t =  time()                                      
        print("Running doc counts per time per journal...")
        total_counts_list = []
        labels_list = []
        columns_list = []
        for journal in df_dict:
            df1, df2 = mu.doc_topics_per_time(mallet_model.model, mallet_model.nlp_data, df=df_dict[journal], data_column="title_abstract",
                year_column="year", year_res=5, year_start=1980)
            #df1.to_csv('reports/main_model/journals/dnp5y_{}.csv'.format(journal), index=False)
            #df2.to_csv('reports/main_model/journals/dwp5y_{}.csv'.format(journal), index=False)
            x_val = list(range(1980,2020, 5))
            '''
            mu.plot_doc_topics_per_time(df1, 40, 8, ylabel='Proportion of Documents', xlabel='Years', 
                                    fig_save_path='reports/main_model/journals/p_n_abs_p5y_{}.png'.format(journal), 
                                    x_val=x_val, hide_x_val=False, xtick_space=10, xmintick_space=5, 
                                    custom_titles=mu.MAIN_TOPICS_TRUNC, relative_val=True,
                                    df_data2=df1, relative_val2=False, ylabel2="Absolute Count of Documents", show=False)
            '''
            columns = list(df1.columns)[1:]
            column_totals = df1.loc[:,columns[0]:].sum(axis=0)
            column_totals_list = list(column_totals)
            columns_list.append(list(columns))
            total_counts_list.append(column_totals_list)
            labels_list.append(journal)
        legend_params = {'loc':2, 'fontsize':'xx-small'}
        mu.graph_multi(columns_list, total_counts_list, label_list=labels_list, title="Total Abstracts by Year", 
                x_label="Years", y_label="Number of Abstracts", legend="Journal", legend_params=legend_params,
                fig_save_path='reports/main_model/journals/total_docs_per_5year.png', show=True)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time) 

    if False: # looking at long tokens
        long_token_list = []
        for doc in raw_text:
            long_token_list.extend(gensim.utils.simple_preprocess(str(doc), deacc=True, min_len=16, max_len=50))
        print(len(long_token_list))
        long_token_set = set(long_token_list)
        
        print(long_token_set)
        maxlen = 0
        for token in long_token_set:
            if len(token) > maxlen:
                maxlen = len(token)
        print("max length: {}".format(maxlen))

        nlp = spacy.load(mallet_model.nlp_data.spacy_lib, disable=['parser','ner'])
        allowed_postags = ['NOUN', 'ADJ', 'VERB','ADV']

        doc = nlp(" ".join(long_token_list))
        lem_list = [token.lemma_ for token in doc if token.pos_ in allowed_postags and token.lemma_ not in ['-PRON-']
                    ]
        print(len(lem_list))
        lem_set = set(lem_list)
        print(len(lem_set))
        print(lem_set)

    if False: # Testing Alternate Model
        with open('models/main_mallet_t40a5o200_v2', 'rb') as model:
            mallet_model_v2 = pickle.load(model)

        t =  time()                                      
        print("Creating sample colored paragraph ...")
        for doc in [0, 7650, 11442, 14261, 18651]:
            mu.color_doc_topics(mallet_model_v2.model, raw_text[doc], mallet_model_v2.nlp_data, topics=3, line_word_length=12,
                        fig_save_path='reports/v2/sample_colordoctopicsa5_{}.png'.format(doc), show=False, custom_titles=mu.MAIN_TOPICS_V2,
                        incl_perc=True)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if True: # Loading alternate models
        with open('models/main_mallet_t40a5o200_v3', 'rb') as model:
            mallet_model_2 = pickle.load(model)
        with open('models/main_mallet_t40a25o200_v3', 'rb') as model:
            mallet_model_3 = pickle.load(model)

    if False: # Creating wordclouds alternate model
        t =  time()                                      
        print("Creating topic word clouds ...")
        mu.creat_multi_wordclouds(40, 8, mallet_model_v2.model, mallet_model_v2.nlp_data, custom_titles=mu.MAIN_TOPICS_V2, show=False,
            fig_save_path='reports/v2/topic_wordclouds5a.png', title_font=14, seed=2020)
        comp_time =  (time() - t)   
        print("Done in %0.3fs." % comp_time)

    if False: # Creating trends for alternate model
        t =  time()                                      
        print("Running doc counts per time ...")
        df5, df6 = mu.doc_topics_per_time(mallet_model_v2.model, mallet_model_v2.nlp_data, year_list=year_list, year_res=5)
        df5.to_csv('reports/v2/doc_n_per5_year.csv', index=False)
        df6.to_csv('reports/v2/doc_w_per5_year.csv', index=False)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)    

        t =  time()                                      
        print("Creating graph of number of docs per topic per year...")
        x_val = list(range(1980,2020, 5))
        mu.plot_doc_topics_per_time(df5, 40, 8, ylabel='Proportion of Documents', xlabel='Years', fig_save_path='reports/v2/plot_n_abs_per5y.png', 
                                    x_val=x_val, hide_x_val=False, xtick_space=10, xmintick_space=5, custom_titles=mu.MAIN_TOPICS_V2, relative_val=True,
                                    df_data2=df5, relative_val2=False, ylabel2="Absolute Count of Documents", show=False)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Create several colored paragraph samples
        t =  time()                                      
        print("Creating sample colored paragraph ...")
        model_list = [(mallet_model, mu.MAIN_TOPICS_TRUNC, 'm1'), (mallet_model_2, mu.MAIN_TOPICS_V2,'m2'), (mallet_model_3, mu.MAIN_TOPICS_V3, 'm3')]
        for i in range(20):
            index = random.randrange(20000)
            for model_data in model_list:
                model = model_data[0]
                topic_names = model_data[1]
                append = model_data[2]
                mu.color_doc_topics(model.model, raw_text[index], model.nlp_data, topics=10, max_chars=120, incl_perc=True,
                    fig_save_path='reports/v3/color_doc_comp/doc_{}{}.png'.format(index, append), show=False, min_phi=1, 
                    topic_names=topic_names, highlight=True)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False:
        index = 11097
        t =  time()                                      
        print("Creating sample colored paragraph ...")
        model_list = [(mallet_model, mu.MAIN_TOPICS_TRUNC, 'm1'), (mallet_model_2, mu.MAIN_TOPICS_V2,'m2'), (mallet_model_3, mu.MAIN_TOPICS_V3, 'm3')]
        model_list = [(mallet_model_2, mu.MAIN_TOPICS_V2,'m2')]
        indexes = [7118, 10891, 7536, 13422, 6775]
        if len(indexes) == 0:
            for i in range(20):
                indexes.append(random.randrange(20000))
        
        for index in indexes:
            for model_data in model_list:
                model = model_data[0]
                topic_names = model_data[1]
                append = model_data[2]
                mu.color_doc_topics(model.model, raw_text[index], model.nlp_data, topics=10, max_chars=120, incl_perc=True,
                    fig_save_path='reports/doc_candidates/doc_{}{}.png'.format(index, append), show=False, min_phi=1, 
                    topic_names=topic_names, highlight=True)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False:
        dataframes = mfp.reload_dfs('reports/main_a5/data/', 5)
        path = 'reports/main_a5/journals/journal_stats.csv'
        journals = pd.read_csv(path)    
        dataframes['topics_summary'].astype(int, errors='ignore').to_latex('reports/main_a5/topic_w_kws.tex', index=False)
        journals.astype(int, errors='ignore').to_latex('reports/main_a5/journal_stats.tex', index=False)
        
    if False:
        df_dict = mfp.reload_dfs('reports/main_a5/data/', 5)
        output_folder_path = 'reports/main_a5/figures/'
        topic_names = mu.MAIN_TOPICS_V2
        years = list(range(1980,2020,5))
        plt_param = {
            'n_topics':40,
            'n_horiz':8,
            'xlabel':'Years',
            'ylabel':'Proportion of Documents', 
            'ylabel2':"Absolute Count of Documents",
            'xtick_space':10, 
            'xmintick_space':5
        }
        mu.plot_doc_topics_per_time(df_dict['docs/topic/year'],  
            fig_save_path=output_folder_path + 'ndom.png', 
            x_val=years, hide_x_val=False, topic_names=topic_names, relative_val=True,
            df_data2=df_dict['docs/topic/year'], relative_val2=False, show=False, **plt_param
            )

        plt_param['ylabel'] = "Proportional Weight of Topic"
        mu.plot_doc_topics_per_time(df_dict['doc_weight/topic/year'],  
            fig_save_path=output_folder_path + 'weights.png', 
            x_val=years, hide_x_val=False, topic_names=topic_names, relative_val=True,
            df_data2=df_dict['docs/topic/year'], relative_val2=False, show=False, **plt_param
            )

    if False:
        data_path = 'data/external/data_cleaned.csv'
        data_column = 'title_abstract'

        df = pd.read_csv(data_path)
        data = df[data_column].tolist()

        nlp_params = dict(spacy_lib='en_core_sci_lg', max_df=.25, bigrams=True, trigrams=True, max_tok_len=30)
        nlp_data = data_nl_processing_v2.NlpForLdaInput(data, **nlp_params)
        nlp_data.start()
        with open('models/nl_data', 'wb') as file:            
            pickle.dump(nlp_data, file)
        mu.plot_doc_token_counts(nlp_data=nlp_data)

    if False:
        with open('models/nl_data', 'rb') as model:
            nlp_data = pickle.load(model)
        mu.plot_doc_token_counts(nlp_data=nlp_data, bins=None)

    if False:
        data_path = 'data/external/data_cleaned.csv'
        data_column = 'title_abstract'

        df = pd.read_csv(data_path)
        data = df[data_column].tolist()

        nlp_params = dict(spacy_lib='en_core_sci_lg', max_df=.25, bigrams=True, trigrams=True, max_tok_len=30)
        nlp_data = data_nl_processing_v2.NlpForLdaInput(data, **nlp_params)
        nlp_data.start()
        seed = 83747765
        coh = 'c_v'

        model1 = mu.MalletModel(nlp_data, topics=40, seed=seed, coherence=coh, **{'alpha':25, 'optimize_interval':200})
        model2 = mu.MalletModel(nlp_data, topics=40, seed=seed, coherence=coh, **{'alpha':25, 'optimize_interval':0})
        model3 = mu.MalletModel(nlp_data, topics=40, seed=seed, coherence=coh, **{'alpha':50, 'optimize_interval':0})
        model4 = mu.MalletModel(nlp_data, topics=40, seed=seed, coherence=coh, model_type='gensim')
        model1.start()
        model2.start()
        model3.start()
        model4.start()
        coh_list = [
            [model1.model_raw['coherence']],[model2.model_raw['coherence']],[model3.model_raw['coherence']],[model4.model_raw['coherence']]
        ]
        coh_setting = [
            {'coherence':'c_v', 'window':10},
            {'coherence':'c_uci'}, {'coherence':'c_npmi'}, {'coherence':'u_mass'}
        ]
        models = [model1, model2, model3, model4]
        for coh_set in coh_setting:
            for num, model in enumerate(models):
                coh_list[num].append(coherence_set(model, **coh_set))

        print("Coherence for optimized: C_V:{}, C_V_w10:{}, C_UCI:{}, C_NPMI:{}, UMASS:{}".format(*coh_list[0]))
        print("Coherence for unoptimized: C_V:{}, C_V_w10:{}, C_UCI:{}, C_NPMI:{}, UMASS:{}".format(*coh_list[1]))
        print("Coherence for a50: C_V:{}, C_V_w10:{}, C_UCI:{}, C_NPMI:{}, UMASS:{}".format(*coh_list[2]))
        print("Coherence for gensim: C_V:{}, C_V_w10:{}, C_UCI:{}, C_NPMI:{}, UMASS:{}".format(*coh_list[3]))
    if True:
        dataframes = mfp.reload_dfs('reports/main_a5/data/', 5)

        path = 'reports/main_a5/journals/journal_stats.csv'
        groups_path1 = 'reports/main_a5/data/grouped_topics_5y_lr.csv'
        groups_path2 = 'reports/main_a5/data/grouped_topics_total.csv'
        journals = pd.read_csv(path)
        groups_lr =  pd.read_csv(groups_path1)
        groups_t = pd.read_csv(groups_path2)
        topics = dataframes['topics_summary']
        topics.columns = ["Topic", "Name", "Keywords", "Document Count", "Coefficient*10^3", "R^2"] 
        topics["Coefficient*10^3"] = topics["Coefficient*10^3"] * 1000

        groups_t['Document Weight'] = groups_t['Document Weight'] * 100
        groups_t["Coefficient*10^3"] = groups_lr["Coefficient"] * 1000
        groups_t["R^2"] = groups_lr["R^2"]
        groups_t.columns = ['Topic Group', 'Topics', 'Percentage of Documents', 'Coefficient*10^3', 'R^2']

        topics = topics.round(decimals={'Coefficient*10^3':2, 'R^2':2 })
        groups_t = groups_t.round(decimals={'Percentage of Documents':2, 'Coefficient*10^3':2, 'R^2':2 })
        with pd.option_context("max_colwidth", 1000):
            topics.to_latex('reports/main_a5/topics_summary.tex', index=False)
            journals.to_latex('reports/main_a5/journal_stats.tex', index=False)
            groups_t.to_latex('reports/main_a5/groups_summary.tex', index=False)
