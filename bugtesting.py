
if __name__ == "__main__":
    from time import time
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import re, nltk, gensim, spacy
    import scispacy
    import pickle
    from nltk.corpus import stopwords

    #from lemmatization_methods import nltk_lemmatizer

    #Gensim
    import gensim
    from gensim.corpora.dictionary import Dictionary

    #sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
    from sklearn.model_selection import GridSearchCV
    from pprint import pprint
    import data_nl_processing

    print("Modules loaded")

    data_path = 'data/processed/data_methods_split.csv'
    #import dataset
    print("Loading dataset for bug testing...")
    t0 = time()
    df = pd.read_csv(data_path)
    data = df["title_abstract"].tolist()
    print("done in %0.3fs." % (time() - t0))

    nlp_data = data_nl_processing.NlpForLdaInput(data)
    nlp_data.start()

    # Building the Gensim LDA model
    print("Building Gensim LDA model...")
    t0 = time()

    corpus_gensim = nlp_data.gensim_lda_input()
    id2word_gensim = nlp_data.get_id2word()

    print("corpus and id2word have been assigned")
    gensim_lda = gensim.models.LdaMulticore(corpus=corpus_gensim,
                                id2word=id2word_gensim,
                                num_topics=40,
                                decay=0.5,
                                workers=5,
                                random_state=100,
                                chunksize=100,
                                passes=10,
                                per_word_topics=True)

    print("done in %0.3fs." % (time() - t0))

    gensim_topics = gensim_lda.show_topics(formatted=False, num_words=10, num_topics=-1)

    def print_top_words_gensim(show_topics):
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

    gensim_topics_list = print_top_words_gensim(gensim_topics)

    gensim_coh_model = gensim.models.CoherenceModel(topics=gensim_topics_list, texts=nlp_data.get_token_text(),
                                                    dictionary=nlp_data.get_id2word(), window_size=None,
                                                    coherence='c_v')
    gensim_coherence = gensim_coh_model.get_coherence()
    print("Gensim Coherence: {:0.3f}".format(gensim_coherence))