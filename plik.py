from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords, reuters
from sklearn.feature_extraction.text import TfidfVectorizer
cachedStopWords = stopwords.words("english")

def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words if word not in cachedStopWords]
    tokens =(list(map(lambda token: PorterStemmer().stem(token),words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens));
    return filtered_tokens

def tf_idf(docs):
    tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3,max_df=0.90, max_features=3000,use_idf=True, sublinear_tf=True,norm='l2');
    tfidf.fit(docs);
    return tfidf;

def feature_values(doc, representer):
    doc_representation = representer.transform([doc])
    features = representer.get_feature_names()
    return [(features[index], doc_representation[0, index]) for index in doc_representation.nonzero()[1]]


def main():
    train_docs = []
    test_docs = []

    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            train_docs.append(reuters.raw(doc_id))
        else:
            test_docs.append(reuters.raw(doc_id))

    representer = tf_idf(train_docs);

    for doc in test_docs:
        print(feature_values(doc, representer))

main()