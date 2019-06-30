import spacy
import re

def lexical(text, duration=10):
    categories = ['i', 'we', 'they', 'DET', 'VERB', 'ADV',
                    'prep', 'conj', 'neg', 'PDT', 'NUM']
    features_dict = {name:0 for name in categories}
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for token in doc:
        if(token.text.lower() in features_dict.keys()):
            features_dict[token.text.lower()] +=1
        if(token.dep_ in features_dict.keys()):
            features_dict[token.dep_] += 1
        if(token.pos_ in features_dict.keys()):
            features_dict[token.pos_] += 1
    punc_pattern = re.compile("\W+")
    text = re.sub(punc_pattern, " ", text)
    words_list = text.lower().split()
    uwords_list = list(set(list(words_list)))
    wpsec = len(words_list)/duration
    upsec = len(uwords_list)/duration
    features_dict.update([('wpsec', wpsec), ('upsec', upsec)])
    return features_dict.values()
