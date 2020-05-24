if __name__ == "__main__":
    from time import time
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import pickle
    import gensim, data_nl_processing, data_nl_processing_v2

    import glob

    #plotting tools
    import pyLDAvis
    import pyLDAvis.gensim
    import matplotlib.pyplot as plt
    import model_utilities

    if False:
        data_path = 'data/processed/data_methods_split.csv'
        data_column_t = 'title_abstract'
        data_column_m = 'methods'
        print("Loading dataset for CompareModels testing...")
        t0 = time()
        df = pd.read_csv(data_path)
        data_t = df[data_column_t].tolist()
        data_m = df[data_column_m].tolist()
        print("done in %0.3fs." % (time() - t0))

        spacy_library = 'en_core_sci_lg'
        nlp_data_t = data_nl_processing.NlpForLdaInput(data_t, spacy_lib=spacy_library, max_df=.25, bigrams=True, trigrams=True)
        nlp_data_t.start()

        nlp_data_m = data_nl_processing.NlpForLdaInput(data_m, spacy_lib=spacy_library, max_df=.25, bigrams=True, trigrams=True)
        nlp_data_m.start()

        model_seed = int(time()*100)-158000000000

        model_t = model_utilities.MalletModel(nlp_data_t, topics=20, seed=model_seed, model_type='mallet')
        model_t.start()
        model_t.save('models/mallet_nomethods{}'.format(model_seed))

        model_m = model_utilities.MalletModel(nlp_data_m, topics=25, seed=model_seed, model_type='mallet')
        model_m.start()
        model_m.save('models/mallet_methods{}'.format(model_seed))

    if False:

        with open('models/mallet_methods393847761', 'rb') as model:
            mallet_model_m = pickle.load(model)
        with open('models/mallet_nomethods393847761', 'rb') as model:
            mallet_model_t = pickle.load(model)


        panel = pyLDAvis.gensim.prepare(mallet_model_t.model, mallet_model_t.nlp_data.gensim_lda_input(), mallet_model_t.nlp_data.get_id2word(), 
                                        mds='tsne', sort_topics=False)
        pyLDAvis.save_html(panel, 'reports/figures/pylda_mallet_nomethods393847761.html')

        panel = pyLDAvis.gensim.prepare(mallet_model_m.model, mallet_model_m.nlp_data.gensim_lda_input(), mallet_model_m.nlp_data.get_id2word(), 
                                        mds='tsne', sort_topics=False)
        pyLDAvis.save_html(panel, 'reports/figures/pylda_mallet_methods393847761.html')

    if False:
        data_path = 'data/external/data_cleaned.csv'
        data_column = 'title_abstract'
        print("Loading dataset for CompareModels testing...")
        t0 = time()
        df = pd.read_csv(data_path)
        data = df[data_column].tolist()
        print("done in %0.3fs." % (time() - t0))

        spacy_library = 'en_core_sci_lg'
        nlp_data = data_nl_processing.NlpForLdaInput(data, spacy_lib=spacy_library, max_df=.25, bigrams=True, trigrams=True)
        nlp_data.start()

        topic_num = 40
        model_seed = int(time()*100)-158000000000

        model_def = model_utilities.MalletModel(nlp_data, topics=topic_num, seed=model_seed, model_type='mallet')
        model_def.start()
        model_def.save('models/mallet_def_{}T{}'.format(topic_num, model_seed))

        model_opt = model_utilities.MalletModel(nlp_data, topics=topic_num, seed=model_seed, model_type='mallet', optimize_interval=100)
        model_opt.start()
        model_opt.save('models/mallet_opt100_{}T{}'.format(topic_num, model_seed))
        
    if False:
        with open('models/mallet_def_40T396683912', 'rb') as model:
            model_def = pickle.load(model)
        with open('models/mallet_opt100_40T396683912', 'rb') as model:
            model_opt = pickle.load(model)

        panel_def = pyLDAvis.gensim.prepare(model_def.model, model_def.nlp_data.gensim_lda_input(), model_def.nlp_data.get_id2word(), 
                                        mds='tsne', sort_topics=False)
        pyLDAvis.save_html(panel_def, 'reports/figures/pylda_mallet_def_{}T{}.html'.format(model_def.topics, model_def.seed))

        panel_opt = pyLDAvis.gensim.prepare(model_opt.model, model_opt.nlp_data.gensim_lda_input(), model_opt.nlp_data.get_id2word(), 
                                        mds='tsne', sort_topics=False)
        pyLDAvis.save_html(panel_opt, 'reports/figures/pylda_mallet_opt100_{}T{}.html'.format(model_opt.topics, model_opt.seed))

    if False:
        data_path = 'data/external/data_cleaned.csv'
        data_column = 'title_abstract'
        print("Loading dataset for CompareModels testing...")
        t0 = time()
        df = pd.read_csv(data_path)
        data = df[data_column].tolist()
        print("done in %0.3fs." % (time() - t0))

        spacy_library = 'en_core_sci_lg'
        nlp_data = data_nl_processing.NlpForLdaInput(data, spacy_lib=spacy_library, max_df=.25, bigrams=True, trigrams=True)
        nlp_data.start()

        topic_num = 40
        model_seed = int(time()*100)-158000000000

        model_mal = model_utilities.MalletModel(nlp_data, topics=topic_num, seed=model_seed, model_type='mallet', 
                                        alpha=25, optimize_interval=200)
        model_mal.start()
        model_mal.save('models/mallet_malopt_{}T{}'.format(topic_num, model_seed))

        model_gen = model_utilities.MalletModel(nlp_data, topics=topic_num, seed=model_seed, model_type='gensim', 
                                        alpha='asymmetric', eta='auto')
        model_gen.start()
        model_gen.save('models/mallet_genopt_{}T{}'.format(topic_num, model_seed))

        print('Mallet coherence is: {}'.format(model_mal.model_raw['coherence']))
        print('Gensim coherence is: {}'.format(model_gen.model_raw['coherence']))

        panel_mal = pyLDAvis.gensim.prepare(model_mal.model, model_mal.nlp_data.gensim_lda_input(), model_mal.nlp_data.get_id2word(), 
                                        mds='tsne', sort_topics=False)
        pyLDAvis.save_html(panel_mal, 'reports/figures/pylda_mallet_malopt_{}T{}.html'.format(model_mal.topics, model_mal.seed))

        panel_gen = pyLDAvis.gensim.prepare(model_gen.model, model_gen.nlp_data.gensim_lda_input(), model_gen.nlp_data.get_id2word(), 
                                        mds='tsne', sort_topics=False)
        pyLDAvis.save_html(panel_gen, 'reports/figures/pylda_mallet_genopt_{}T{}.html'.format(model_gen.topics, model_gen.seed))

    if False:
        data_path = 'data/external/data_cleaned.csv'
        data_column = 'title_abstract'
        print("Loading dataset for main model building...")
        t0 = time()
        df = pd.read_csv(data_path)
        data = df[data_column].tolist()
        print("done in %0.3fs." % (time() - t0))

        spacy_library = 'en_core_sci_lg'
        nlp_data = data_nl_processing.NlpForLdaInput(data, spacy_lib=spacy_library, max_df=.25, bigrams=True, trigrams=True)
        nlp_data.start()

        topic_num = 40
        model_seed = int(time()*100)-158000000000
        models = [{'alpha':5,'optimize_interval':0}, 
                    {'alpha':5,'optimize_interval':200},
                    {'alpha':10,'optimize_interval':0},
                    {'alpha':10,'optimize_interval':200},
                    {'alpha':25,'optimize_interval':0},
                    {'alpha':25,'optimize_interval':200},
                    {'alpha':50,'optimize_interval':0},
                    {'alpha':50,'optimize_interval':200}]
        model_list = []
        for i in range(len(models)):
            model = model_utilities.MalletModel(nlp_data, topics=topic_num, seed=model_seed, model_type='mallet', **models[i])
            model.start()
            save_path = 'models/main_mallet_t{}a{}o{}'.format(topic_num, models[i]['alpha'], models[i]['optimize_interval'])
            model.save(save_path)
            model_list.append((model, save_path))

        with open('reports/main_mallet_parameters_{}T.txt'.format(topic_num), 'w') as para_file:
            file_string_list = []
            file_string_list.append("Main Model Parameters for {} Topics \n".format(topic_num))
            file_string_list.append("\n")
            for i in range(len(model_list)):
                file_string_list.append("Mallet model t{}a{}o{} Parameters: \n".format(topic_num, models[i]['alpha'], models[i]['optimize_interval']))
                file_string_list.append("Model {}/{} generated \n".format(i+1, len(model_list)))
                file_string_list.append("File Path: {} \n".format(model_list[i][1]))
                file_string_list.append("{} Coherence: {} \n".format(model_list[i][0].coherence, model_list[i][0].model_raw['coherence']))
                for key in model_list[i][0].parameters:
                    file_string_list.append("{}: {} \n".format(key, model_list[i][0].parameters[key]))
                file_string_list.append("\n")
            para_file.writelines(file_string_list)

        for i in range(len(model_list)):
            save_path = "reports/figures/main_mallet_t{}a{}o{}s{}.html".format(
                            topic_num, models[i]['alpha'], models[i]['optimize_interval'], model_seed)
            panel = pyLDAvis.gensim.prepare(model_list[i][0].model, model_list[i][0].nlp_data.gensim_lda_input(), model_list[i][0].nlp_data.get_id2word(), 
                                        mds='tsne', sort_topics=False)
            pyLDAvis.save_html(panel, save_path)

    if False:
        model_file_list = glob.glob('.\\models\\main_mallet_t*')
        for model_path in model_file_list:
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
                save_path = "reports/figures/main_mallet_t{}a{}o{}s{}.html".format(
                            model.topics, model.parameters['alpha'], model.parameters['optimize_interval'], model.seed)
                panel = pyLDAvis.gensim.prepare(model.model, model.nlp_data.gensim_lda_input(), 
                                                model.nlp_data.get_id2word(), mds='tsne', sort_topics=False)
                pyLDAvis.save_html(panel, save_path)

    if False:
        model_file_list = glob.glob('.\\models\\main_mallet_t*')
        for model_path in model_file_list:
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
                save_path = "reports/figures/main_mallet_t{}a{}o{}s{}_wc.png".format(
                            model.topics, model.parameters['alpha'], model.parameters['optimize_interval'], model.seed)
                model_utilities.creat_multi_wordclouds(40, 7, model.model, model.nlp_data, num_w=20, fig_dpi=400,
                            show=False, fig_save_path=save_path)

    if False:
        data_path = 'data/external/data_cleaned.csv'
        data_column = 'title_abstract'
        print("Loading dataset for main model building...")
        t0 = time()
        df = pd.read_csv(data_path)
        data = df[data_column].tolist()
        print("done in %0.3fs." % (time() - t0))

        spacy_library = 'en_core_sci_lg'
        nlp_data = data_nl_processing.NlpForLdaInput(data, spacy_lib=spacy_library, max_df=.25, bigrams=True, trigrams=True, max_tok_len=30)
        nlp_data.start()

        topic_num = 40
        model_seed = int(time()*100)-158000000000
        models = [{'alpha':5,'optimize_interval':0}, 
                    {'alpha':5,'optimize_interval':200},
                    {'alpha':10,'optimize_interval':0},
                    {'alpha':10,'optimize_interval':200},
                    {'alpha':25,'optimize_interval':0},
                    {'alpha':25,'optimize_interval':200},
                    {'alpha':50,'optimize_interval':0},
                    {'alpha':50,'optimize_interval':200}]
        model_list = []
        for i in range(len(models)):
            model = model_utilities.MalletModel(nlp_data, topics=topic_num, seed=model_seed, model_type='mallet', **models[i])
            model.start()
            save_path = 'models/main_mallet_t{}a{}o{}_v2'.format(topic_num, models[i]['alpha'], models[i]['optimize_interval'])
            model.save(save_path)
            model_list.append((model, save_path))

        with open('reports/v2/main_mallet_parameters_{}T_v2.txt'.format(topic_num), 'w') as para_file:
            file_string_list = []
            file_string_list.append("Main Model Parameters for {} Topics \n".format(topic_num))
            file_string_list.append("\n")
            for i in range(len(model_list)):
                file_string_list.append("Mallet model t{}a{}o{} Parameters: \n".format(topic_num, models[i]['alpha'], models[i]['optimize_interval']))
                file_string_list.append("Model {}/{} generated \n".format(i+1, len(model_list)))
                file_string_list.append("File Path: {} \n".format(model_list[i][1]))
                file_string_list.append("{} Coherence: {} \n".format(model_list[i][0].coherence, model_list[i][0].model_raw['coherence']))
                for key in model_list[i][0].parameters:
                    file_string_list.append("{}: {} \n".format(key, model_list[i][0].parameters[key]))
                file_string_list.append("\n")
            para_file.writelines(file_string_list)

        for i in range(len(model_list)):
            save_path = "reports/v2/main_mallet_t{}a{}o{}s{}_v2.html".format(
                            topic_num, models[i]['alpha'], models[i]['optimize_interval'], model_seed)
            panel = pyLDAvis.gensim.prepare(model_list[i][0].model, model_list[i][0].nlp_data.gensim_lda_input(), model_list[i][0].nlp_data.get_id2word(), 
                                        mds='tsne', sort_topics=False)
            pyLDAvis.save_html(panel, save_path)

        for i in range(len(model_list)):
            save_path = "reports/v2/main_mallet_t{}a{}o{}s{}_wc_v2.png".format(
                            topic_num, models[i]['alpha'], models[i]['optimize_interval'], model_seed)
            model_utilities.creat_multi_wordclouds(40, 8, model_list[i][0].model, model_list[i][0].nlp_data, num_w=20, fig_dpi=400,
                            show=False, fig_save_path=save_path)

    if False: # Docs by dominant topic
        with open('models/main_mallet_t40a25o200_v2', 'rb') as model:
            mallet_model = pickle.load(model)
        t =  time()                                      
        print("Running docs by dominant topic ...")
        topic_df = model_utilities.dominant_doc_topic_df(mallet_model.model, mallet_model.nlp_data)    
        topic_df.to_csv('reports/v2/docs_dom_topic.csv')
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False: # Document token histogram
        t =  time()                                      
        print("Creating doc token counts ...")
        model_utilities.plot_doc_token_counts(topic_df,fig_save_path='reports/v2/doc_token_counts.png', show=False)
        comp_time =  (time() - t)                                       
        print("Done in %0.3fs." % comp_time)

    if False:
        data_path = 'data/external/data_cleaned.csv'
        data_column = 'title_abstract'
        print("Loading dataset for main model building...")
        t0 = time()
        df = pd.read_csv(data_path)
        data = df[data_column].tolist()
        print("done in %0.3fs." % (time() - t0))

        spacy_library = 'en_core_sci_lg'
        nlp_data = data_nl_processing_v2.NlpForLdaInput(data, spacy_lib=spacy_library, max_df=.25, bigrams=True, trigrams=True, max_tok_len=30)
        nlp_data.start()

        topic_num = 40
        model_seed = 629740313
        models = [{'alpha':5,'optimize_interval':0}, 
                    {'alpha':5,'optimize_interval':200},
                    {'alpha':10,'optimize_interval':0},
                    {'alpha':10,'optimize_interval':200},
                    {'alpha':25,'optimize_interval':0},
                    {'alpha':25,'optimize_interval':200},
                    {'alpha':50,'optimize_interval':0},
                    {'alpha':50,'optimize_interval':200}]
        model_list = []
        for i in range(len(models)):
            model = model_utilities.MalletModel(nlp_data, topics=topic_num, seed=model_seed, model_type='mallet', **models[i])
            model.start()
            save_path = 'models/main_mallet_t{}a{}o{}_v3'.format(topic_num, models[i]['alpha'], models[i]['optimize_interval'])
            model.save(save_path)
            model_list.append((model, save_path))

        with open('reports/v3/main_mallet_parameters_{}T_v3.txt'.format(topic_num), 'w') as para_file:
            file_string_list = []
            file_string_list.append("Main Model Parameters for {} Topics \n".format(topic_num))
            file_string_list.append("\n")
            for i in range(len(model_list)):
                file_string_list.append("Mallet model t{}a{}o{} Parameters: \n".format(topic_num, models[i]['alpha'], models[i]['optimize_interval']))
                file_string_list.append("Model {}/{} generated \n".format(i+1, len(model_list)))
                file_string_list.append("File Path: {} \n".format(model_list[i][1]))
                file_string_list.append("{} Coherence: {} \n".format(model_list[i][0].coherence, model_list[i][0].model_raw['coherence']))
                for key in model_list[i][0].parameters:
                    file_string_list.append("{}: {} \n".format(key, model_list[i][0].parameters[key]))
                file_string_list.append("\n")
            para_file.writelines(file_string_list)

        for i in range(len(model_list)):
            save_path = "reports/v3/main_mallet_t{}a{}o{}s{}_v3.html".format(
                            topic_num, models[i]['alpha'], models[i]['optimize_interval'], model_seed)
            panel = pyLDAvis.gensim.prepare(model_list[i][0].model, model_list[i][0].nlp_data.gensim_lda_input(), model_list[i][0].nlp_data.get_id2word(), 
                                        mds='tsne', sort_topics=False)
            pyLDAvis.save_html(panel, save_path)

        for i in range(len(model_list)):
            save_path = "reports/v3/main_mallet_t{}a{}o{}s{}_wc_v3.png".format(
                            topic_num, models[i]['alpha'], models[i]['optimize_interval'], model_seed)
            model_utilities.create_multi_wordclouds(40, 8, model_list[i][0].model, model_list[i][0].nlp_data, num_w=20, fig_dpi=400,
                            show=False, fig_save_path=save_path)
    
    if True:
        data_path = 'data/external/data_cleaned.csv'
        data_column = 'title_abstract'
        print("Loading dataset for main model building...")
        t0 = time()
        df = pd.read_csv(data_path)
        data = df[data_column].tolist()
        print("done in %0.3fs." % (time() - t0))

        spacy_library = 'en_core_sci_lg'
        nlp_data = data_nl_processing_v2.NlpForLdaInput(data, spacy_lib=spacy_library, max_df=.25, bigrams=True, trigrams=True, max_tok_len=30)
        nlp_data.start()

        topic_num = 20
        model_seed = 629740313
        models = [{'alpha':5,'optimize_interval':200},
                    {'alpha':25,'optimize_interval':200},
                    {'alpha':50,'optimize_interval':200}]
        model_list = []
        for i in range(len(models)):
            model = model_utilities.MalletModel(nlp_data, topics=topic_num, seed=model_seed, model_type='mallet', **models[i])
            model.start()
            save_path = 'models/main_mallet_t{}a{}o{}_v3'.format(topic_num, models[i]['alpha'], models[i]['optimize_interval'])
            model.save(save_path)
            model_list.append((model, save_path))

        with open('reports/v3/main_mallet_parameters_{}T_v3.txt'.format(topic_num), 'w') as para_file:
            file_string_list = []
            file_string_list.append("Main Model Parameters for {} Topics \n".format(topic_num))
            file_string_list.append("\n")
            for i in range(len(model_list)):
                file_string_list.append("Mallet model t{}a{}o{} Parameters: \n".format(topic_num, models[i]['alpha'], models[i]['optimize_interval']))
                file_string_list.append("Model {}/{} generated \n".format(i+1, len(model_list)))
                file_string_list.append("File Path: {} \n".format(model_list[i][1]))
                file_string_list.append("{} Coherence: {} \n".format(model_list[i][0].coherence, model_list[i][0].model_raw['coherence']))
                for key in model_list[i][0].parameters:
                    file_string_list.append("{}: {} \n".format(key, model_list[i][0].parameters[key]))
                file_string_list.append("\n")
            para_file.writelines(file_string_list)

        for i in range(len(model_list)):
            save_path = "reports/v3/main_mallet_t{}a{}o{}s{}_v3.html".format(
                            topic_num, models[i]['alpha'], models[i]['optimize_interval'], model_seed)
            panel = pyLDAvis.gensim.prepare(model_list[i][0].model, model_list[i][0].nlp_data.gensim_lda_input(), model_list[i][0].nlp_data.get_id2word(), 
                                        mds='tsne', sort_topics=False)
            pyLDAvis.save_html(panel, save_path)

        for i in range(len(model_list)):
            save_path = "reports/v3/main_mallet_t{}a{}o{}s{}_wc_v3.png".format(
                            topic_num, models[i]['alpha'], models[i]['optimize_interval'], model_seed)
            model_utilities.create_multi_wordclouds(20, 5, model_list[i][0].model, model_list[i][0].nlp_data, num_w=20, fig_dpi=400,
                            show=False, fig_save_path=save_path)