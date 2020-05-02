from time import time
from datetime import datetime

import numpy as np
import pandas as pd
import re, gensim, spacy
import scispacy
import nltk
from nltk.corpus import stopwords

#from lemmatization_methods import nltk_lemmatizer

#sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from pprint import pprint

#plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LdaAnalysis:
    def __init__(self, model, vectorizer, data):
        self.model = model
        self.vectorizer = vectorizer
        self.data = data
        self.vectorized_data = vectorizer.transform(data)
        self.model_output = self.model.transform(self.vectorized_data)       
        self.df_doc_topic = None
        self.df_topic_distribution = None

    def print_top_words(self, model=None, feature_names=None, n_top_words=20):
        if model is None: model = self.model
        if feature_names is None: feature_names = self.vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += ", ".join([feature_names[i]
                                for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()

    def print_topic_distribution(self):
        if self.df_doc_topic is None:
            self.build_topics_doc_dataframe(print_table=False, j_display_table=False)
        self.df_topic_distribution = self.df_doc_topic['Dominant Topic'].value_counts().reset_index(name='Num Documents')
        self.df_topic_distribution.columns = ['Topic Num', 'Num Documents']
        print(self.df_topic_distribution)
        return self.df_topic_distribution

    def build_topics_doc_dataframe(self, print_table=True, top=10, j_display_table=True): # Returns a dataframe of topics for each document
        # Note: the index + 1 is there so that the first entry is 1 as opposed to 0, this helps syncronize with pyLDAvis topic output
        # Column names
        topic_names = ["Topic " + str(i+1) for i in range(self.model.n_components)]
        # Index names
        doc_names = ["Doc " + str(i+1) for i in range(len(self.data))]
        # Make dataframe
        self.df_doc_topic = pd.DataFrame(np.round(self.model_output, 2), columns=topic_names, index=doc_names)
        # Dominant topic for each document
        dominant_topic = np.argmax(self.df_doc_topic.values, axis=1)+1
        self.df_doc_topic['Dominant Topic'] = dominant_topic

        # Display table with styling
        if print_table:
            print(self.df_doc_topic[0:top])
        if j_display_table:
            df_doc_topics_table = self.df_doc_topic.head(top).style.applymap(self.color_green_).applymap(self.make_bold_)
            df_doc_topics_table
        return self.df_doc_topic

    def visualize_lda_model(self): # Only works within Ipython/Jupyter
        pyLDAvis.enable_notebook()
        panel = pyLDAvis.sklearn.prepare(self.model, self.vectorized_data, self.vectorizer, mds='tsne')
        panel

    def save_interactive_html(self, html_path):
        panel = pyLDAvis.sklearn.prepare(self.model, self.vectorized_data, self.vectorizer, mds='tsne', sort_topics=False)
        pyLDAvis.save_html(panel, html_path)

    def return_top_word_df(self, n_top_words=20, print_top=True):
        topic_keywords = self.show_topics_(n_top_words)
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word ' + str(i+1) for i in range(len(df_topic_keywords.columns))]
        df_topic_keywords.index = ['Topic ' + str(i+1) for i in range(len(df_topic_keywords.index))]
        if print_top: print(df_topic_keywords)
        return df_topic_keywords

    def plot_doc_cluster(self, scatter3d=False, kmean=True): # needs some troubleshooting with labeling
        # Construct k-means cluster
        if kmean:
            clusters = KMeans(n_clusters=self.model.get_params()['n_components']).fit_predict(self.model_output)
            for i in range(len(clusters)):
                clusters[i] += 1
            legend_title = "Cluster"
        else:
            if self.df_doc_topic is None:
                self.build_topics_doc_dataframe(print_table=False, j_display_table=False)
            clusters = self.df_doc_topic['Dominant Topic'].tolist()
            legend_title = "Topic"
        # Build Singular Value Decomposition (SVD) model
        if scatter3d:
            num_comp=3
        else:
            num_comp=2

        svd_model = TruncatedSVD(n_components=num_comp)
        lda_output_svd = svd_model.fit_transform(self.model_output)
        # X and Y axes of the plot using SVD decomposition
        x = lda_output_svd[:,0]
        y = lda_output_svd[:,1]
        if scatter3d: z = lda_output_svd[:,2]
        # Weights for the columns of lda_output, for each component
        print("Component's weights: \n", np.round(svd_model.components_, 2))
        # Percentage of total information in 'lda_output' explained by the two components
        print("Percentage of variance explained: \n", np.round(svd_model.explained_variance_ratio_, 2))
        # Plot

        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111, projection='3d')
        if scatter3d:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(x, y, z, c=clusters, s=10, alpha=0.5)
            ax.set_zlabel("Component 3")
        else:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(x, y, c=clusters, s=10, alpha=0.5)

        ax.set_ylabel("Component 2")
        ax.set_xlabel("Component 1")
        ax.set_title("Segregation of Topic Clusters")
        label_list = []
        for t in scatter.legend_elements(): # for some reason only 8 entries generated if >12 n_components
            label_list.append(t)
        # Warning1: Kmeans cluster number does not directly correspond to topic number  
        # Warning2: Not sure why but the number of legend plots drops to 8 after n_clusters>12
        legend = ax.legend(handles=label_list[0], labels=label_list[1], loc='best', title=legend_title)
        ax.add_artist(legend)
        plt.show()

        #plt.figure(figsize=(12,12))
        #plt.scatter(x, y, c=clusters)
        #plt.ylabel("Component 2")
        #plt.xlabel("Component 1")
        #fig.title("Segregation of Topic Clusters")
        #plt.show()

    def predict_topic(self, text, spacy_lem_lib='en_core_sci_md' ):
        # Text Preprocessing
        nlp = spacy.load(spacy_lem_lib, disable=['parser','ner'])
        text_words = list(gensim.utils.simple_preprocess(str(text),deacc=True)) #deacc=True removes punctuations
        text_tokens = nlp(" ".join(text_words))
        lem_text = " ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in text_tokens])
        # Text Vectorization
        text_vec = self.vectorizer.transform([lem_text])
        # LDA Transform
        topic_probability_scores = self.model.transform(text_vec)
        topic = np.argmax(topic_probability_scores) + 1 # The + 1 is because I made the topics start at 1 instead of 0
        return topic, topic_probability_scores

    def get_similar_docs(self, text, top_docs=5, spacy_lem_lib='en_core_sci_md', verbose=False):
        topic, lda_output = self.predict_topic(text, spacy_lem_lib)
        dists = euclidean_distances(lda_output.reshape(1, -1), self.model_output)[0]
        doc_ids = np.argsort(dists)[:top_docs]
        if verbose:
            print("Topic number: ", topic)
            print("Topic probability scores: ", np.round(lda_output, 1))
            print("Most similar document's probability scores: ", np.round(self.model_output[doc_ids], 1))
        return doc_ids, np.take(self.data, doc_ids)

    def show_topics_(self, n_top_words):
        keywords = np.array(self.vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in self.model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_top_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    def color_green_(self, val):
        color = 'green' if val > .1 else 'black'
        return 'color: {col}'.format(col=color)

    def make_bold_(self, val):
        weight = 700 if val > .1 else 400
        return 'font-weight: {weight}'.format(weight=weight)

if __name__ == "__main__": # Prevents the following code from running when importing module
    #import dataset

    max_rows = 1000
    num_components = 10
    data_column = 'title_abstract'
    n_gram_max = 1

    print("Loading dataset...")
    t0 = time()
    df = pd.read_csv('data/external/data_methods_split.csv')
    data = df[data_column].tolist()
    print("done in %0.3fs." % (time() - t0))

    print("Preprocessing dataset...")
    print('Creating Tokens...')
    t0 = time()

    #process dataset:
    #Tokenize
    def sent_to_words(sentences):
        for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence),deacc=True)) #deacc=True removes punctuations
    data_words = list(sent_to_words(data[:max_rows]))
    print("done in %0.3fs." % (time() - t0))

    #Lemmatization using spaCy package for lemmatization (simpler than NLTK)
    #https://spacy.io/api/annotation
    print('Lemmetization in progress using spaCy...')
    t0 = time()
    nlp = spacy.load('en_core_sci_md', disable=['parser','ner'])
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB','ADV']):
        output_text = []
        for abstract in texts:
                doc = nlp(" ".join(abstract))
                output_text.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc]))
        return output_text
                
    data_lemmatized_spacy = lemmatization(data_words)
    print("done in %0.3fs." % (time() - t0))
    #print(data_lemmatized_spacy[:1])

    # Vectorizing dataset for use with LDA algorithm
    print('Vectorizing dataset...')
    t0 = time()
    stop_words = set(stopwords.words('english'))
    stop_words.update(['elsevier', 'copyright', 'rights', 'reserve', 'reserved', 'ed'])
    vectorizer = CountVectorizer(analyzer='word',               # features are 'word' vs 'char' or 'char_wb' (char_wb or only characters inside words)
                            min_df=10,                          # minimum number of word repeats
                            stop_words=stop_words,               # removes stop words (can also use max_df)
                            lowercase=True,                     # converts everything to lowercase
                            token_pattern='[a-zA-Z0-9]{2,}',    # char considered and max length
                            ngram_range=(1,n_gram_max),                   # n-gram range (min, max)
                            #max_features=50000,                # Max number of features (words from highest freq to lowest)
                            max_df=0.5                        # Removes words that occur in more than n proportion of texts
                            )

    data_vectorized = vectorizer.fit_transform(data_lemmatized_spacy)
    print(vectorizer)
    #checking sparcity because it was in my tutorial (is this an important thing?)
    data_dense = data_vectorized.todense()
    print('Sparcity: ', ((data_dense > 0).sum()/data_dense.size)*100, '%')
    print("done in %0.3fs." % (time() - t0))

    # Building the LDA model
    print('Building LDA model...')
    t0 = time()
    lda_model = LatentDirichletAllocation(n_components = num_components,              # Number of topics
                                        max_iter=10,                  # Max learning iterations
                                        learning_method='online',     # Learning method batch versus online
                                        random_state=100,             # Random state seed
                                        batch_size=128,               # Batch size for online learning
                                        evaluate_every=-1,            # Calculates perplexity every n iterations (it is off)
                                        n_jobs=-1,                    # Number of CPUs used (-1 = all)
                                        )

    lda_output = lda_model.fit_transform(data_vectorized)
    print(lda_model) # Prints model attributes
    print("done in %0.3fs." % (time() - t0))

    # Printing top 20 words for each topic
    n_top_words = 20
    visualizer = LdaAnalysis(lda_model, vectorizer, data_lemmatized_spacy)
    visualizer.print_top_words()
    
    visualizer.print_topic_distribution()
    visualizer.build_topics_doc_dataframe()

    visualizer.return_top_word_df(15)
    print(visualizer.predict_topic(data[0]))
    print(visualizer.get_similar_docs(data[0], top_docs=2, verbose=True))

    visualizer.plot_doc_cluster()
    #visualizer.plot_doc_cluster(kmean=False)
    #visualizer.plot_doc_cluster(scatter3d=True)

    now = datetime.now().strftime("_%m%d%Y%H%M")
    #visualizer.save_interactive_html('reports/figures/SKL_LDA{}.html'.format(now))
