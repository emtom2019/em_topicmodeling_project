#%%
from time import time
from datetime import datetime
import os, sys

import numpy as np
import pandas as pd
import pickle
import gensim, data_nl_processing
import model_utilities
import spacy
import scispacy
from collections import OrderedDict

import glob

#plotting tools
import math
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud


# %%
if __name__ == "__main__":
    with open('models/main_mallet_t40a25o200', 'rb') as model:
        mallet_model = pickle.load(model)
    
    data_path = 'data/external/data_cleaned.csv'
    data_column = 'title_abstract'
    df = pd.read_csv(data_path)
    raw_text = df[data_column].tolist()
    
    model_utilities.plot_tsne_doc_cluster(mallet_model.model, mallet_model.nlp_data)

# %%
    model = mallet_model.model
    doc = raw_text[0]
    nlp_data = mallet_model.nlp_data
    line_word_length=10 
    show=True
    fig_save_path=None 
    topics=5
    fig_save_path='reports/figures/testing_colordoctopics.png'

#%%

    colors = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    if topics > 10:
        topics = 10
    doc_prep = gensim.utils.simple_preprocess(str(doc), deacc=True, min_len=2)
# %%
    doc_split = str(doc).split('.')
    new_raw = []
    for sentence in doc_split:
        sentence_tok = gensim.utils.simple_preprocess(str(sentence), deacc=True,                            min_len=1)
        if len(sentence_tok) > 0:
            sentence_tok[-1] += '.'
            new_raw += sentence_tok
    print(len(new_raw))
# %%
    doc_raw = gensim.utils.simple_preprocess(str(doc), deacc=True, min_len=1)

    wordset = set(doc_raw)
    doc_index_dict = {}
    for word in wordset:
        word_indexes = [i for i, w in enumerate(doc_raw) if w == word]
        doc_index_dict[word] = word_indexes
    
    token_index_dict = {}
    token_list = []

    nlp = spacy.load(nlp_data.spacy_lib, disable=['parser','ner'])
    allowed_postags = ['NOUN', 'ADJ', 'VERB','ADV']

    for word in doc_prep:
        if word not in nlp_data.stopwords:
            token = nlp(word)[0]
            if token.pos_ in allowed_postags and token.lemma_ not in ['-PRON-']:
                token_list.append(token.lemma_)
                if token.lemma_ in token_index_dict:
                    token_index_dict[token.lemma_] += doc_index_dict[word]
                else:
                    token_index_dict[token.lemma_] = doc_index_dict[word]
    for token in token_index_dict:
        token_index_dict[token] = sorted(set(token_index_dict[token]))

    processed_tokens = nlp_data.process_ngrams_([token_list])[0]
    final_token_dict = {}
    for token in processed_tokens:
        if token not in final_token_dict:
            final_token_dict[token] = []
        split_tokens = token.split('_')
        for split_token in split_tokens:
            final_token_dict[token].append(token_index_dict[split_token].pop(0))
#%%
    
    lines = math.ceil(len(doc_raw) / line_word_length)
    fig, axes = plt.subplots(lines, 1, figsize=(line_word_length, lines), dpi=150)
    axes[0].axis('off')
    
#%%
    topic_perc, wordid_topics, wordid_phivalues = model.get_document_topics(
        nlp_data.gensim_lda_input([" ".join(processed_tokens)])[0], per_word_topics=True)
    topic_perc_sorted = sorted(topic_perc, key=lambda x:(x[1]), reverse=True)
    top_topics = [topic[0] for i, topic in enumerate(topic_perc_sorted) if i < topics]
    top_topics_color = {top_topics[i]:i for i in range(len(top_topics))}

#%%
    word_dom_topic = {}
    for wd, wd_topics in wordid_topics:
        for topic in wd_topics:
            if topic in top_topics:
                word_dom_topic[model.id2word[wd]] = topic
                break

#%%
    index_color_dict = {}
    for token in final_token_dict:
        if token in word_dom_topic:
            for i in final_token_dict[token]:
                index_color_dict[i] = top_topics_color[word_dom_topic[token]]

# %%
    n = line_word_length
    doc_raw_lines = [doc_raw[i * n:(i + 1) * n] for i in range(lines)]

# %%
    for i, ax in enumerate(axes):
        if i > 0:
            word_pos = 0.06
            for index in range(len(doc_raw_lines[i-1])):
                word = doc_raw_lines[i-1][index]
                raw_index = index + (i - 1) * n 
                if raw_index in index_color_dict:
                    color = colors[index_color_dict[raw_index]]
                else:
                    color = 'black'
                ax.text(word_pos, 0.5, word, horizontalalignment='left',
                            verticalalignment='center', fontsize=16, color=color,
                            transform=ax.transAxes, fontweight=700)
                word_pos += .009 * len(word)  # to move the word for the next iter
                ax.axis('off')



# %%
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Document colored by top {} topics'.format(topics), 
                    fontsize=22, y=0.95, fontweight=700)
    plt.tight_layout()
    plt.show()                

# %%
