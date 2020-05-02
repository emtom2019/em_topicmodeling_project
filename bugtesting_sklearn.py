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
    print("done in %0.3fs." % (time() - t0))

    spacy_library = 'en_core_sci_lg'

    nlp_data = data_nl_processing.NlpForLdaInput(data, spacy_lib=spacy_library, max_df=0.25, bigrams=False, trigrams=True)
    nlp_data.start()

    # Building the Gensim LDA model

    corpus_gensim = nlp_data.gensim_lda_input()
    id2word_gensim = nlp_data.get_id2word()
    mallet_path='C:\\mallet\\bin\\mallet'
    print('Building SKL LDA model...')
    t0 = time()
    # default coherence is .477
    lda_model = LatentDirichletAllocation(n_components = 20,          # Number of topics
                                        doc_topic_prior=0.06,         # default alpha None
                                        topic_word_prior= 0.04,        # default beta None
                                        learning_decay=.7,
                                        max_iter=10,                  # Max learning iterations
                                        learning_method='online',     # Learning method batch versus online
                                        random_state=100,             # Random state seed
                                        batch_size=128,               # Batch size for online learning
                                        evaluate_every=-1,            # Calculates perplexity every n iterations (it is off)
                                        n_jobs=-1,                    # Number of CPUs used (-1 = all)
                                        )

    corpus_sklearn = nlp_data.sklearn_lda_input()
    lda_output = lda_model.fit_transform(corpus_sklearn)
    print("done in %0.3fs." % (time() - t0))

    tf_feature_names = nlp_data.get_feature_names()  

    def print_top_words_skl(model, feature_names, n_top_words):
        topic_word_list = []
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            new_list = list(feature_names[i]
                                for i in topic.argsort()[:-n_top_words - 1:-1])
            message += ", ".join(new_list)
            topic_word_list.append(new_list)
            print(message)
        print()
        return topic_word_list
    n_top_words = 20
    skl_topics_list = print_top_words_skl(lda_model, tf_feature_names, n_top_words) 

    skl_coh_model = gensim.models.CoherenceModel(topics=skl_topics_list, texts=nlp_data.get_token_text(),
                                                    dictionary=nlp_data.get_id2word(), window_size=None,
                                                    coherence='c_v')
    skl_coherence = skl_coh_model.get_coherence()
    print("Skl Coherence: {:0.3f}".format(skl_coherence))