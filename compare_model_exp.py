# must have following line in windows 10 when using multicore gensim, otherwise
# the entire code gets rerun by the number of workers and it gets out of control
#%%
if __name__ == "__main__":

    print("Importing Modules")
    from time import time
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import re, gensim, spacy
    import scispacy
    import pickle
    import nltk
    from nltk.corpus import stopwords

    #from lemmatization_methods import nltk_lemmatizer

    #sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
    from sklearn.model_selection import GridSearchCV, KFold
    from pprint import pprint

    #plotting tools
    import pyLDAvis
    import pyLDAvis.sklearn
    import matplotlib.pyplot as plt

    import data_nl_processing

    mallet_path = 'C:\\mallet\\bin\\mallet'
    data_path = 'data/external/data_methods_split.csv'
    #import dataset
    print("Loading dataset for comparing models...")
    t0 = time()
    df = pd.read_csv(data_path)
    data = df["title_abstract"].tolist()
    print("done in %0.3fs." % (time() - t0))


    nlp_data = data_nl_processing.NlpForLdaInput(data)
    nlp_data.start()

    # %%
    # Building the SKLearn LDA model
    print('Building SKL LDA model...')
    t0 = time()
    lda_model = LatentDirichletAllocation(n_components = 10,              # Number of topics
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

    # Printing top 10 words for each topic
    n_top_words = 10

    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += ", ".join([feature_names[i]
                                for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()

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

    skl_topics_list = print_top_words_skl(lda_model, tf_feature_names, n_top_words) 
    

#%%
    # Building the Gensim LDA model
    print("Building Gensim LDA model...")
    t0 = time()
    corpus_gensim = nlp_data.gensim_lda_input()
    id2word_gensim = nlp_data.get_id2word()
    gensim_lda = gensim.models.LdaMulticore(corpus=corpus_gensim,
                                id2word=id2word_gensim,
                                num_topics=10,
                                decay=0.7,
                                workers=5,
                                random_state=100,
                                chunksize=100,
                                passes=10,
                                per_word_topics=True)

    print("done in %0.3fs." % (time() - t0))



    gensim_topics = gensim_lda.show_topics(formatted=False, num_words=10)

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

    # %%
    # Building the Mallet LDA model
    print("Building Mallet LDA model...")
    t0 = time()
    #import os
    #os.environ['MALLET_HOME'] = 'C:\\mallet\\'
    mallet_path = 'C:\\mallet\\bin\\mallet'
    mallet_lda = gensim.models.wrappers.LdaMallet(mallet_path=mallet_path,
                                corpus=corpus_gensim,
                                id2word=id2word_gensim,
                                num_topics=10,
                                workers=5,
                                random_seed=100,
                                iterations=1000)


    print("done in %0.3fs." % (time() - t0))

    mallet_topics = mallet_lda.show_topics(formatted=False, num_words=10)
    mallet_topics_list = print_top_words_gensim(mallet_topics)

    # %%
    # Building coherence model from gensim
    lem_text = nlp_data.get_lem_text()
    texts = []
    for doc in lem_text:
        texts.append(doc.split())
    
    coherence_model_umass = gensim.models.CoherenceModel(topics=skl_topics_list, texts=texts, dictionary=nlp_data.get_id2word(), coherence='u_mass', topn=10)
    coherence_model_cv = gensim.models.CoherenceModel(topics=skl_topics_list, texts=texts, dictionary=nlp_data.get_id2word(), coherence='c_v', topn=10)


# %%
    print("U_mass Coherence Score: ", coherence_model_umass.get_coherence())
    print("C_V Coherence Score: ", coherence_model_cv.get_coherence())


# %%
    print("U_mass coherence score: ") 
    pprint(coherence_model_umass.compare_model_topics([gensim_topics_list, skl_topics_list]))

    print("C_V coherence score: ") 
    pprint(coherence_model_cv.compare_model_topics([gensim_topics_list, skl_topics_list]))

# %%
    coherence_model_umass2 = gensim.models.CoherenceModel(topics=gensim_topics_list, texts=texts, dictionary=nlp_data.get_id2word(), coherence='u_mass', topn=10)
    coherence_model_cv2 = gensim.models.CoherenceModel(topics=gensim_topics_list, texts=texts, dictionary=nlp_data.get_id2word(), coherence='c_v', topn=10)

#%%
    print("Gensim U_mass Coherence Score: ", 
        coherence_model_umass2.get_coherence())
    print("Gensim C_V Coherence Score: ", 
        coherence_model_cv2.get_coherence())



# %%
    coherence_model_umass3 = gensim.models.CoherenceModel(topics=mallet_topics_list, 
                            texts=texts, dictionary=nlp_data.get_id2word(), coherence='u_mass', topn=10)
    coherence_model_cv3 = gensim.models.CoherenceModel(topics=mallet_topics_list, 
                            texts=texts, dictionary=nlp_data.get_id2word(), coherence='c_v', topn=10)
    print("Mallet U_mass Coherence Score: ", coherence_model_umass3.get_coherence())
    print("Mallet C_V Coherence Score: ", coherence_model_cv3.get_coherence())
