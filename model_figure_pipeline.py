from time import time
from datetime import datetime
import os, sys

import numpy as np
import pandas as pd
import gensim, spacy, scispacy
import pickle
import glob
import random
import re

import data_nl_processing
import model_utilities as mu

def load_model(path):
    with open(path, 'rb') as model:
        mallet_model = pickle.load(model)
    return mallet_model

def load_data(path, data_column, year_column):
    df = pd.read_csv(path)
    df.fillna(int(0), inplace=True)
    raw_text = df[data_column].tolist()
    df[year_column] = df[year_column].apply(lambda x: int(x) if isinstance(x, int) or isinstance(x, float) else 0).to_list()
    year_list = df[year_column].tolist()
    return df, raw_text, year_list

def build_dfs(output_folder_path, model, raw_text, year_list, year_res, topic_names=None, topic_groups=None, year_start=None, year_end=None):
    # Generate dataframes for number and weights of topics total and per year_res number of years
    df_nt, df_wt = mu.docs_per_topic(model.model, model.nlp_data)
    df_nty, df_wty = mu.doc_topics_per_time(
        model.model, model.nlp_data, year_list=year_list, year_res=year_res, year_start=year_start, year_end=year_end
        )
 
    # Save above dataframes
    df_nty.to_csv(output_folder_path + 'doc_n_per_topic{}y.csv'.format(year_res), index=False)
    df_wty.to_csv(output_folder_path + 'doc_w_per_topic{}y.csv'.format(year_res), index=False)
    df_nt.to_csv(output_folder_path + 'doc_n_per_topic.csv', index=False)
    df_wt.to_csv(output_folder_path + 'doc_w_per_topic.csv', index=False)
    # Generate and save dataframe of docs with their dominant topic
    df_domtop = mu.dominant_doc_topic_df(model.model, model.nlp_data)    
    df_domtop.to_csv(output_folder_path + 'docs_dom_topic.csv', index=False)

    # Generate and save dataframes of the best document for each topic, with tokens and with raw text
    df_bestdoc = mu.best_doc_for_topic(df_domtop)
    df_bestdoc.to_csv(output_folder_path + 'best_doc_per_topic.csv', index=False)
    doc_list = df_bestdoc["Best Document"]
    new_column = []
    for doc in doc_list:
        new_column.append(raw_text[int(doc-1)])
    df_bestdoc_raw = df_bestdoc.copy()
    df_bestdoc_raw["Raw Text"] = pd.Series(new_column).values
    df_bestdoc_raw.to_csv(output_folder_path + 'best_doc_per_topic_with_raw.csv', index=False)
    # Build dataframe with topic number, +- topic name, Topic keywords, and total document count
    df_summary = mu.build_summary_df(df_bestdoc, df_nt, df_wty, topic_names=topic_names, rel_val=True)
    df_summary.to_csv(output_folder_path + 'topics_summary.csv', index=False)

    # Generate and save topic co-occurence matrix
    df_cooc_matrix, df_cooc_n = mu.build_cooc_matrix_df(model.model, model.nlp_data)
    df_cooc_matrix.to_csv(output_folder_path + 'topics_cooc_matrix.csv', index=True)
    df_cooc_n.to_csv(output_folder_path + 'topics_cooc_matrix_num.csv', index=True)

    # Save dataframes for linear regression - These are not returned
    if topic_groups is not None:
        df_grouped, df_lr= mu.plot_topic_groups(df_wty, topic_groups, show=False, linear_reg=True)
        df_totals = mu.plot_topic_groups(df_wt, topic_groups, show=False)
        df_grouped.to_csv(output_folder_path + 'grouped_topics_{}y.csv'.format(year_res), index=False)
        df_lr.to_csv(output_folder_path + 'grouped_topics_{}y_lr.csv'.format(year_res), index=False)
        df_totals.to_csv(output_folder_path + 'grouped_topics_total.csv', index=False)
    # Returns all dataframes as a dictionary
    dataframe_dict = {
        'docs/topic':df_nt, 'doc_weight/topic':df_wt, 'docs/topic/year':df_nty, 
        'doc_weight/topic/year':df_wty, 'dom_topic/doc':df_domtop, 
        'best_doc/topic':df_bestdoc, 'best_doc/topic_w_raw':df_bestdoc_raw,
        'topics_summary':df_summary, 'cooc_matrix':df_cooc_matrix
        }
    return dataframe_dict
    
def reload_dfs(output_folder_path, year_res):
    # Loads all data frames that were generated by build_dfs function and returns them as a dictionary
    df_nty = pd.read_csv(output_folder_path + 'doc_n_per_topic{}y.csv'.format(year_res))
    df_wty = pd.read_csv(output_folder_path + 'doc_w_per_topic{}y.csv'.format(year_res))
    df_nt = pd.read_csv(output_folder_path + 'doc_n_per_topic.csv')
    df_wt = pd.read_csv(output_folder_path + 'doc_w_per_topic.csv')
    df_domtop = pd.read_csv(output_folder_path + 'docs_dom_topic.csv')
    df_bestdoc = pd.read_csv(output_folder_path + 'best_doc_per_topic.csv')
    df_bestdoc_raw = pd.read_csv(output_folder_path + 'best_doc_per_topic_with_raw.csv')
    df_summary = pd.read_csv(output_folder_path + 'topics_summary.csv')    
    df_cooc_matrix = pd.read_csv(output_folder_path + 'topics_cooc_matrix.csv', index_col=0)
    dataframe_dict = {
        'docs/topic':df_nt, 'doc_weight/topic':df_wt, 'docs/topic/year':df_nty, 
        'doc_weight/topic/year':df_wty, 'dom_topic/doc':df_domtop, 
        'best_doc/topic':df_bestdoc, 'best_doc/topic_w_raw':df_bestdoc_raw,
        'topics_summary':df_summary, 'cooc_matrix':df_cooc_matrix
        }
    return dataframe_dict

def build_figures_all_data(output_folder_path, model, df_dict, raw_text, year_res, year_list, topic_names=None,
                            topic_groups=None, year_start=None, year_end=None, tsne_seed=2020, pic_format='png', **kwargs):
    # Builds figures based on model and dataframe
    # Build doc token count histogram
    mu.plot_doc_token_counts(
        nlp_data=model.nlp_data, fig_save_path=output_folder_path + 'doc_token_counts.{}'.format(pic_format), show=False
        )
    # Build graph of numbers of docs per year range
    columns = list(df_dict['docs/topic/year'].columns)[1:]
    column_totals = df_dict['docs/topic/year'].loc[:,columns[0]:].sum(axis=0)
    column_totals_list = list(column_totals)
    plt_param2 = {
        'title':'Total Abstracts by Year',
        'x_label':'Years',
        'y_label':'Number of Abstracts', 
    }
    for key in kwargs:
        if key in plt_param2:
            plt_param2[key] = kwargs[key]
    mu.graph(
        columns, column_totals_list, 
        fig_save_path=output_folder_path + 'total_docs_per_{}year.{}'.format(year_res, pic_format),
        **plt_param2
        )
    total_docs = 0
    for total in column_totals_list:
        total_docs += total
    print("Total docs: {}".format(total_docs))
    # Graph number of docs per topic per year - 
    sorted_years = sorted(year_list)
    if year_start is None:
        year_start = sorted_years[0]
    if year_end is None:
        year_end = sorted_years[-1]
    years = list(range(year_start, year_end+1, year_res))
    plt_param = {
        'n_topics':model.model.num_topics,
        'n_horiz':8,
        'xlabel':'Years',
        'ylabel':'Proportion of Literature', 
        'ylabel2':"Absolute Count of Documents",
        'xtick_space':10, 
        'xmintick_space':5
    }
    for key in kwargs:
        if key in plt_param:
            plt_param[key] = kwargs[key]

    mu.plot_doc_topics_per_time(df_dict['doc_weight/topic/year'],  
        fig_save_path=output_folder_path + 'relw_abs_docs_per_t{}y.{}'.format(year_res, pic_format), 
        x_val=years, hide_x_val=False, topic_names=topic_names, relative_val=True,
        df_data2=df_dict['docs/topic/year'], relative_val2=False, show=False, **plt_param
        )
    mu.plot_doc_topics_per_time(df_dict['doc_weight/topic/year'],  
        fig_save_path=output_folder_path + 'relw_docs_per_t{}y.{}'.format(year_res, pic_format), 
        x_val=years, hide_x_val=False, topic_names=topic_names, relative_val=True,
        show=False, **plt_param
        )

    # If topic groups is given, plot trends for groups and save df of linear regression
    if topic_groups is not None:
        plt_gparams = {
            'n_horiz':5,
            'xlabel':'Years',
            'ylabel':'Proportion of Documents', 
            'xtick_space':10, 
            'xmintick_space':5
        }
        mu.plot_topic_groups(df_dict['doc_weight/topic/year'], topic_groups, x_val=years, hide_x_val=False, merge_graphs=True, 
            fig_save_path=output_folder_path + 'topic_groups_merged{}y.{}'.format(year_res, pic_format), show=False, **plt_gparams)
        mu.plot_topic_groups(df_dict['doc_weight/topic/year'], topic_groups, x_val=years, hide_x_val=False, merge_graphs=False, 
            fig_save_path=output_folder_path + 'topic_groups{}y.{}'.format(year_res, pic_format), show=False, **plt_gparams)

    # Create word clouds
    mu.create_multi_wordclouds(plt_param['n_topics'], plt_param['n_horiz'], model.model, model.nlp_data, 
            topic_names=topic_names, show=False, title_font=14, seed=tsne_seed,
            fig_save_path=output_folder_path + 'topic_wordclouds.{}'.format(pic_format))
    
    # Create t-SNE doc cluster graph
    mu.plot_tsne_doc_cluster(
        model.model, model.nlp_data, marker_size=1, min_tw=None, seed=tsne_seed, show_topics=True, 
        show=False, topic_names=topic_names, show_legend=True, 
        fig_save_path=output_folder_path + 'tsne_doc_cluster_s{}.{}'.format(tsne_seed, pic_format),
        size=8
        )
    # Create the cluster heatmap from the co-occurence matrix
    mu.plot_clusterheatmap(df_dict['cooc_matrix'], fig_save_path=output_folder_path + 'topic_cluster_heatmap.{}'.format(pic_format),
            topic_names=topic_names,
            show=False,
            figsize=(12,12),
            dendrogram_ratio=(.05, .2),
            cmap='YlOrRd')

def build_sample_paragraph(output_folder_path, model, raw_text, doc_ids, num_topics, topic_names=None, 
                            max_chars=120, min_phi=1.0, pic_format='png'):
    if isinstance(doc_ids, int):
        doc_ids = [doc_ids]
    for doc_id in doc_ids:
        mu.color_doc_topics(
            model.model, raw_text[doc_id], model.nlp_data, topics=num_topics, max_chars=max_chars, 
            incl_perc=True, fig_save_path=output_folder_path + 'doc_{}.{}'.format(doc_id, pic_format), 
            show=False, topic_names=topic_names, min_phi=min_phi
            )

def build_journal_df_figs(output_folder_path, model, df, journal_column, data_column, year_column, year_res, year_list, 
                        pic_format='png', topic_names=None, topic_groups=None, year_start=None, year_end=None, max_journals=6, **kwargs):
    # Breaks down the dataframe into smaller dataframes per journal
    df_dict, counts = mu.rows_per_df_grp(df, journal_column)
    # Creates a dataframe for journal, first year published, and total abstracts per journal
    count_dict = dict(counts)
    journal_list = []
    firsty_list = []
    count_list = []
    for journal in df_dict:
        y_list = df_dict[journal]['year'].tolist()
        y_list.sort()
        journal_list.append(journal)
        for y in y_list:
            if y != 0:
                firsty_list.append(y)
                break
            firsty_list.append(y)
        count_list.append(count_dict[journal])
    journal_dict = {
        'Journal':journal_list,
        'First Publication':firsty_list,
        'Number of Abstracts':count_list
    }
    df_journal = pd.DataFrame(journal_dict)
    df_journal.to_csv(output_folder_path + 'journal_stats.csv', index=False)
    # If the Number of Journals is greater than max_journals, the top max_journals are displayed
    # And the rest are places together in the 'Other Journals' group
    if len(journal_list) > max_journals:
        # This remakes the Journal Stats file with the other column
        sorted_df = df_journal.sort_values(['Number of Abstracts'], ascending=False)
        firsty = np.min(sorted_df.iloc[max_journals:]['First Publication'])
        other = 'Other Journals'
        num_abstracts = np.sum(sorted_df.iloc[max_journals:]['Number of Abstracts'])
        other_data = [{
        'Journal':other,
        'First Publication':firsty,
        'Number of Abstracts':num_abstracts
        }]
        other_row = pd.DataFrame(other_data)
        top_journals = sorted_df.head(max_journals).sort_values(['Journal'])
        new_df_journal = pd.concat([top_journals, other_row], ignore_index=True)
        new_df_journal.to_csv(output_folder_path + 'journal_stats_other.csv', index=False)

        # This rebuilds the df_dict to have an 'Other Journals' Group
        top_journals_list = top_journals['Journal'].to_list()
        new_df_dict = {}
        other_df = pd.DataFrame()
        for journal in df_dict:
            if journal in top_journals_list:
                new_df_dict[journal] = df_dict[journal]
            else:
                other_df = pd.concat([other_df, df_dict[journal]], ignore_index=True)
        new_df_dict[other] = other_df
        df_dict = new_df_dict

    # Creates and saves the individual dataframes per journal with doc number and weight per year per topic
    total_counts_list = []
    labels_list = []
    columns_list = []
    plt_param = {
        'n_topics':model.model.num_topics,
        'n_horiz':8,
        'xlabel':'Years',
        'ylabel':'Proportion of Documents', 
        'ylabel2':"Absolute Count of Documents",
        'xtick_space':10, 
        'xmintick_space':5
    }
    for key in kwargs:
        if key in plt_param:
            plt_param[key] = kwargs[key]

    sorted_years = sorted(year_list)
    if year_start is None:
        year_start = sorted_years[0]
    if year_end is None:
        year_end = sorted_years[-1]
    years = list(range(year_start, year_end+1, year_res))

    for journal in df_dict:
        # Creates Dataframes and saves them
        df1, df2 = mu.doc_topics_per_time(model.model, model.nlp_data, df=df_dict[journal], data_column=data_column,
            year_column=year_column, year_res=year_res, year_start=year_start, year_end=year_end)
        df1.to_csv(output_folder_path + 'dnp{}y_{}.csv'.format(year_res, journal), index=False)
        df2.to_csv(output_folder_path + 'dwp{}y_{}.csv'.format(year_res, journal), index=False)
        
        df_nt, df_wt = mu.docs_per_topic(model.model, model.nlp_data, doc_list=df_dict[journal][data_column].to_list())
        df_nt.to_csv(output_folder_path + 'dnt_{}.csv'.format(journal), index=False)
        df_wt.to_csv(output_folder_path + 'dwt_{}.csv'.format(journal), index=False)
        # creates the plots of topic per time for all topics
        df_lr_all = mu.plot_doc_topics_per_time(df2,
            fig_save_path=output_folder_path + 'rel_abs_docs_per_t{}y_{}.{}'.format(year_res, journal, pic_format), 
            x_val=years, hide_x_val=False, topic_names=topic_names, relative_val=True,
            df_data2=df1, relative_val2=False, show=False, linear_reg=True, **plt_param
            )
        df_lr_all.to_csv(output_folder_path + 'lr_rel_docs_per_t{}y_{}.csv'.format(year_res, journal), index=False)
        # Adds topic group figures and dataframes
        if topic_groups is not None:
            plt_gparams = {
            'n_horiz':5,
            'xlabel':'Years',
            'ylabel':'Proportion of Documents', 
            'xtick_space':10, 
            'xmintick_space':5
            }
            df_grouped, df_lr = mu.plot_topic_groups(df2, topic_groups, x_val=years, hide_x_val=False, merge_graphs=True, 
                fig_save_path=output_folder_path + 'topic_groups_merged{}y_{}.{}'.format(year_res, journal, pic_format), show=False, linear_reg=True, 
                **plt_gparams)
            mu.plot_topic_groups(df2, topic_groups, x_val=years, hide_x_val=False, merge_graphs=False, 
                fig_save_path=output_folder_path + 'topic_groups{}y_{}.{}'.format(year_res, journal, pic_format), show=False, **plt_gparams)
            df_totals = mu.plot_topic_groups(df_wt, topic_groups, show=False)
            df_grouped.to_csv(output_folder_path + 'grouped_topics_{}y_{}.csv'.format(year_res, journal), index=False)
            df_lr.to_csv(output_folder_path + 'grouped_topics_{}y_{}.csv'.format(year_res, journal), index=False)
            df_totals.to_csv(output_folder_path + 'grouped_topics_total_{}.csv'.format(journal), index=False)
        # Saves data to create total abstract counts per journal per time
        columns = list(df1.columns)[1:]
        column_totals = df1.loc[:,columns[0]:].sum(axis=0)
        column_totals_list = list(column_totals)
        columns_list.append(list(columns))
        total_counts_list.append(column_totals_list)
        labels_list.append(journal)
    legend_params = {'loc':2, 'fontsize':'xx-small'}
    plt_param2 = {
        'title':'Total Abstracts by Year',
        'x_label':'Years',
        'y_label':'Number of Abstracts', 
        'legend':'Journal',
        'legend_params':legend_params
    }
    for key in kwargs:
        if key in plt_param2:
            plt_param2[key] = kwargs[key]
    # Graphs total abstract by journals over time
    mu.graph_multi(columns_list, total_counts_list, labels_list, 
        fig_save_path=output_folder_path + 'total_docs_per_{}year.{}'.format(year_res, pic_format), 
        show=False, **plt_param2
        )

def run_pipeline(model_path, data_path, data_column, year_column, journal_column, year_start, year_res, year_end=None, pic_format='png', 
                main_path=None, docs=None, min_phi=1, num_topics=5, seed=2020, topic_names=None, topic_groups=None, max_journals=6):
    # Runs all of the functions and creates all of the figures and dataframes 
    if main_path is None:
        main_path = 'reports/main/'
    path_df = main_path + 'data/'
    path_fig = main_path + 'figures/'
    path_journals = main_path + 'journals/'
    paths = [path_df, path_fig, path_journals]
    # This creates the above directories if they don't exist.
    for path in paths:
        os.makedirs(path, exist_ok=True)

    with mu.Timing("Loading Data..."):
        model = load_model(model_path)
        df, raw_text, year_list = load_data(data_path, data_column, year_column)
    
    with mu.Timing("Creating Dataframes..."):
        df_dict = build_dfs(path_df, model, raw_text, year_list, year_res=year_res, topic_names=topic_names, year_start=year_start,
                            year_end=year_end, topic_groups=topic_groups)
    
    with mu.Timing("Creating Figures..."):
        build_figures_all_data(path_fig, model, df_dict, raw_text, year_res, year_list, topic_groups=topic_groups,
                            topic_names=topic_names, year_start=year_start, year_end=year_end, tsne_seed=seed, 
                            pic_format=pic_format)

    with mu.Timing("Processing data for individual journals..."):
        build_journal_df_figs(path_journals, model, df, journal_column, data_column, year_column, year_res, year_list,
                        topic_names=topic_names, topic_groups=topic_groups, year_start=year_start, year_end=year_end, 
                        max_journals=max_journals, pic_format=pic_format)
    
    with mu.Timing("Creating sample abstract..."):
        if docs is None:
            docs = random.randrange(len(raw_text)-1)
        build_sample_paragraph(path_fig, model, raw_text, docs, num_topics=num_topics, topic_names=topic_names, 
                        max_chars=120, min_phi=min_phi, pic_format=pic_format)


if __name__ == "__main__": # Runs script only if it is directly run
    
    model_path = 'models/main_mallet_t40a25o200_v3'
    data_path = 'data/external/data_cleaned.csv'
    data_column = 'title_abstract'
    year_column = 'year'
    journal_column = 'journal'
    if False:
        with mu.Timing('Running pipline...'):
            run_pipeline(model_path, data_path, data_column, year_column, journal_column, year_start=1980, year_res=5,
                    docs=None, topic_names=mu.MAIN_TOPICS_V3)

    if False:
        with mu.Timing('Running pipline...'):
            model_path = 'models/main_mallet_t40a5o200_v3'
            topic_groups = mu.TOPIC_GROUPS.copy()
            topic_groups.pop('Methods')
            topic_groups.pop('Miscellaneous')
            topic_names = mu.import_topic_names('reports/main_a5/topic_names_trunc.csv')
            run_pipeline(model_path, data_path, data_column, year_column, journal_column, year_start=1980, year_res=5,
                    docs=6545, num_topics=5, topic_names=topic_names, topic_groups=topic_groups, main_path='reports/main_a5/')

    if False:
        dfs = reload_dfs('reports/main_a5/data/', 5)
        topic_groups = mu.TOPIC_GROUPS.copy()
        topic_groups.pop('Methods')
        topic_groups.pop('Miscellaneous')
        years = list(range(1980, 2020, 5))
        plt_param = {
            'n_horiz':5,
            'xlabel':'Years',
            'ylabel':'Proportion of Documents', 
            'xtick_space':10, 
            'xmintick_space':5
        }
        df_lr_all = mu.plot_doc_topics_per_time(dfs['doc_weight/topic/year'], 40, topic_names=mu.MAIN_TOPICS_V2, x_val=years, hide_x_val=False, 
            show=False, linear_reg=True, **plt_param)
        mu.plot_topic_groups(dfs['doc_weight/topic/year'], topic_groups, x_val=years, hide_x_val=False, merge_graphs=True, 
            fig_save_path='reports/main_a5/figures/all_groups_merged.png', show=False, **plt_param)
        df_grouped, df_lr= mu.plot_topic_groups(dfs['doc_weight/topic/year'], topic_groups, x_val=years, hide_x_val=False, merge_graphs=False, 
            fig_save_path='reports/main_a5/figures/all_groups.png', show=False, linear_reg=True, **plt_param)
        df_totals = mu.plot_topic_groups(dfs['doc_weight/topic'], topic_groups, show=False)
        #df_totals.to_csv('reports/main_a5/data/total_groups.csv', index=False)
        df_grouped.to_csv('reports/main_a5/data/groups_time.csv', index=False)
        df_lr_all.to_csv('reports/main_a5/data/doc_weight_time_lr.csv', index=False)
        df_lr.to_csv('reports/main_a5/data/groups_time_lr.csv', index=False)
    if False:
        main_path='reports/main_a5/'
        df_path = main_path + 'data/'
        j_path = main_path + 'journals/'
        topic_groups = mu.TOPIC_GROUPS.copy()
        topic_groups.pop('Methods')
        topic_groups.pop('Miscellaneous')
        years = list(range(1980, 2020, 5))
        
        model = load_model(model_path)
        df, raw_text, year_list = load_data(data_path, data_column, year_column)
        reload_dfs(df_path, 5)
        build_journal_df_figs(j_path, model, df, journal_column, data_column, year_column, 5, year_list,
                        topic_names=mu.MAIN_TOPICS_V2, topic_groups=topic_groups, year_start=1980, year_end=None, 
                        max_journals=6)

    if False:
        model_path = 'models/main_mallet_t40a5o200_v3'
        model = load_model(model_path)
        path = 'reports/main_a5/figures/'
        mu.plot_doc_token_counts(
            nlp_data=model.nlp_data, fig_save_path=path + 'doc_token_counts.png', show=False
        )

    if True:
        plt_param = {
        'n_topics':40,
        'n_horiz':8,
        'xlabel':'Years',
        'ylabel':'Proportion of Literature', 
        'ylabel2':"Absolute Count of Documents",
        'xtick_space':10, 
        'xmintick_space':5
        }
        year_res = 5
        years = list(range(1980, 2020, year_res))
        pic_format = 'png'
        topic_names = mu.import_topic_names('reports/main_a5/topic_names_trunc.csv')
        df_dict = reload_dfs('reports/main_a5/data/', 5)
        output_folder_path = 'reports/main_a5/figures/'
        mu.plot_doc_topics_per_time(df_dict['doc_weight/topic/year'],  
            fig_save_path=output_folder_path + 'relw_abs_docs_per_t{}y.{}'.format(year_res, pic_format), 
            x_val=years, hide_x_val=False, topic_names=topic_names, relative_val=True,
            df_data2=df_dict['docs/topic/year'], relative_val2=False, show=False, **plt_param
            )
        mu.plot_doc_topics_per_time(df_dict['doc_weight/topic/year'],  
            fig_save_path=output_folder_path + 'relw_docs_per_t{}y.{}'.format(year_res, pic_format), 
            x_val=years, hide_x_val=False, topic_names=topic_names, relative_val=True,
            show=False, **plt_param
            )