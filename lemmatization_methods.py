import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

TAG_DICT = {'J':wordnet.ADJ,
            'N':wordnet.NOUN,
            'V':wordnet.VERB,
            'R':wordnet.ADV}

def get_wordnet_pos(word):
    #Maps part of speech for each token for use in WordNetLemmatizer
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return TAG_DICT.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def nltk_lemmatizer(texts):
    output_text = []
    for text in texts:
        text_str = ''
        for word in text:
            #word_s = ps.stem(word)
            text_str += lemmatizer.lemmatize(word, get_wordnet_pos(word)) + ' '
        output_text.append(text_str[:-1])
    return output_text

#print(nltk_lemmatizer([['word','words'],['run','ran','better','best','jogged', 'running']]))
#print(get_wordnet_pos('ran'))