import random
import json
import pickle
import numpy as np

import pythainlp
from pythainlp.augment import WordNetAug

Aug = WordNetAug ()
intents = json.loads(open('intents.json').read())

words= pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.plk','rb'))
model = load_model('chatbot.model')

def clean_up_sentence(sentence):
    sentence_words=pythainlp.word_tokenize(sentence)
    sentence_words=[Aug.aug(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words :
        for i, word in enumerate(words):
            if word == w:
                bag[i] =1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res)if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x:[1],reverse=True)
    return_list = []
    for r in results :
        return_list.append({'intent': classes[r[0]], 'probabiliy':str(r[1])})
    return return_list
