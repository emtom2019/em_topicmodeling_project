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
    from silent_print import SilentPrinting

    print("Modules loaded")

    data_path = 'data/processed/data_methods_split.csv'
    #import dataset
    print("Loading dataset for bug testing...")
    t0 = time()
    df = pd.read_csv(data_path)
    data = df["title_abstract"].tolist()
    data2 = df["methods"].tolist()
    print("done in %0.3fs." % (time() - t0))

    spacy_library = 'en_core_sci_md'

    nlp_data = data_nl_processing.NlpForLdaInput(data, spacy_lib=spacy_library, max_df=0.25, bigrams=True, trigrams=False)
    nlp_data.start() # Coh is 0.529 with 1-grams, 0.511 with bigrams, and 0.490 with bigrams and trigrams

    # Building the Gensim LDA model
    print("Building Gensim LDA model...")
    t0 = time()

    corpus_gensim = nlp_data.gensim_lda_input()
    id2word_gensim = nlp_data.get_id2word()
    mallet_path='C:\\mallet\\bin\\mallet'

    #corpus = nlp_data.gensim_lda_input()
    corpus = nlp_data.process_new_corpus(data2)['gensim']

    print("corpus and id2word have been assigned")
    mallet_lda = gensim.models.wrappers.LdaMallet(mallet_path=mallet_path,
                                                    alpha=25, #50 .608, 25 .640, 20 .623, 15 .588, 10 .631
                                                    corpus=corpus,
                                                    id2word=nlp_data.get_id2word(),
                                                    num_topics=40,
                                                    workers=5,
                                                    random_seed=200,
                                                    optimize_interval=100,# coh=.572 with no optimization, .608 with 100,.597 with 50
                                                    iterations=1000)

    print("done in %0.3fs." % (time() - t0))
    

    gensim_topics = mallet_lda.show_topics(formatted=False, num_words=20, num_topics=-1)


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
    print(nlp_data.get_token_text()[:2])
    mallet_topics_list = print_top_words_gensim(gensim_topics)
    mallet_coh_model = gensim.models.CoherenceModel(topics=mallet_topics_list, texts=nlp_data.get_token_text(),
                                                    dictionary=nlp_data.get_id2word(), window_size=None,
                                                    coherence='c_v')
    mallet_coherence = mallet_coh_model.get_coherence()
    print("Mallet Coherence: {:0.3f}".format(mallet_coherence))