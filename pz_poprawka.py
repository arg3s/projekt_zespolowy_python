# Grupa TI_1 Piotr Maleszczuk, Przemysław Burdelak, Łukasz Błaszczak

from sklearn import svm
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, reuters
import re
import math
from operator import itemgetter
from sklearn.naive_bayes import GaussianNB

cachedStopWords = stopwords.words("english")
min_lenght = 3 #minimalna długość słowa
vocabulary_lenght=300 #długość słownika, jednocześnie długość wektora dokumentu

class corpus:
    def __init__(self):
        self.documents = []
        self.categories = reuters.categories()
        self.tfidf = tf_idf()

        #iteracja po wszystkich dokumentach reuters
        for docid in reuters.fileids():
            #odrzucenie dokumentów, które mają więcej niż 1 kategorię
            if len(reuters.categories(docid)) > 1:
                continue

            #określenie kategorii dokumentu
            cat = 0
            for i in range(90):
                if self.categories[i] in reuters.categories(docid):
                    cat = i

            #określenie czy dokument jest przeznaczony do treningów czy do testów
            if docid.startswith("train"):
                train = 1
            elif docid.startswith("test"):
                train = 0
            else:
                raise()
            text = reuters.raw(docid)
            doc = document(text, cat, train)
            #dodanie dokumentu do klasy TfIdf - potrzebne do późniejszych obliczeń
            self.tfidf.add_document(doc)
            #dodanie dokumentu do tablicy dokumentów
            self.add_document(doc)
        self.initialize_vocabulary()

    def add_document(self, document):
        self.documents.append(document)

    #inicjalizacja słownika
    def initialize_vocabulary(self):

        self.vocabulary = {} #słownik najczęściej występujących słów o strukturze key-value / słowo-numer_słowa_w_słowniku
        self.inverse_vocabulary = {} #odwrócony słownik numer_słowa_w_słowniku-słowo

        self.vocabulary_tfidf = {} #słownik słów z największymi wartościami TfIdf o strukturze key-value / słowo-numer_słowa_w_słowniku
        self.inverse_vocabulary_tfidf = {}

        vocabulary = {} #tymczasowy słownik wszystkich słów o strukturze key-value / słowo-numer_słowa_w_słowniku
        inverse_vocabulary = {} #tymczasowy odwrócony słownik numer_słowa_w_słowniku-słowo

        vocabulary_tfidf = {} #lista sum wartości TfIdf dla danego słowa o strukturze numer_słowa_w_słowniku-wartość_TfIdf
        vocabulary_sizes = {} #lista ilości wystąpień danego słowa w dokumentach o strukturze numer_słowa_w_słowniku-ilość wystąpień

        iterator = 0
        #iteracja po wszystkich dokumentach dodanych do korpusu
        for i, doc in enumerate(self.documents):
            #iteracja po unikalnych słowach dokumentu
            for word in doc.get_unique_words():
                #jesli slowo nie znajduje się jeszcze w słowniku
                if word not in vocabulary.values():
                    #dodanie słowa do słownika
                    vocabulary[iterator] = word
                    inverse_vocabulary[word] = iterator

                    #obliczenie wartości TfIdf dla danego słowa i dokumentu,
                    #a następnie zapisanie wartości w słowniku pod pozycją słowa
                    vocabulary_tfidf[iterator] = self.tfidf.tfidf(word, doc)
                    #ustawienie wartości początkowej
                    vocabulary_sizes[iterator] = 0
                    iterator = iterator + 1

                else: #jeśli słowo znajduje się już w słowniku
                    #zwiększenie ilości wystąpień
                    vocabulary_sizes[inverse_vocabulary[word]] += 1
                    # obliczenie wartości TfIdf dla danego słowa i dokumentu,
                    #a następnie dodanie wartości w słowniku pod pozycją słowa
                    vocabulary_tfidf[inverse_vocabulary[word]] += self.tfidf.tfidf(word, doc)

        #posortowanie wartości TfIdf oraz ilości wystąpień słów malejąco
        sorted_sizes = sorted(vocabulary_sizes.items(), key=itemgetter(1), reverse=True)
        sorted_tfidf = sorted(vocabulary_tfidf.items(), key=itemgetter(1), reverse=True)

        #pobranie n pierwszych wartości z sorted_sizes oraz sorted_tfidf, gdzie n jest zmienną globalną vocabulary_lenght
        keys_sorted = sorted_sizes[:vocabulary_lenght]
        keys_sorted_tfidf = sorted_tfidf[:vocabulary_lenght]

        #wypełnienie ostatecznych słowników
        for i in range(len(vocabulary)):
            #słownik najczęściej występujących słów
            for value in keys_sorted:
                if i == value[0]:
                    self.vocabulary[i] = vocabulary[i]
                    self.inverse_vocabulary[vocabulary[i]] = i
            #słownik słów z największymi wartościami TfIdf
            for value in keys_sorted_tfidf:
                if i == value[0]:
                    self.vocabulary_tfidf[i] = vocabulary[i]
                    self.inverse_vocabulary_tfidf[vocabulary[i]] = i

    #generowanie tablicy wektorów potrzebnych do treningu/testów
    def get_svm_vectors(self, Train=0, Test=0, TfIdf=False):
        Xs = []
        ys = []
        for doc in self.documents:
            #pominięcie dokumentów testowych, jeśli chcemy pobrać te do treningu
            if Train == 1 and doc.train == 0:
                continue
            # pominięcie dokumentów treningowych, jeśli chcemy pobrać te do testów
            if Test == 1 and doc.train == 1:
                continue
            #wybór słownika do utworzenia wektora dokumentu
            if TfIdf:
                x = doc.get_vector(self.inverse_vocabulary_tfidf)
            else:
                x = doc.get_vector(self.inverse_vocabulary)

            y = doc.doc_class
            Xs.append(x)
            ys.append(y)
        return (Xs, ys)

class document:
    def __init__(self, text, doc_class=1, train=1):
        self.doc_class = doc_class
        self.train = train
        self.text = text
        #ze względu na oszczędność czasu preprocessing jest wykonywany tylko raz, podczas tworzenia dokumentu
        self.preprocessed_text = self.preprocessing(self.text.split())

    def preprocessing(self, raw_tokens):
        no_stopwords = [token for token in raw_tokens if token not in cachedStopWords]
        stemmed_tokens = []
        stemmer = PorterStemmer()
        for token in no_stopwords:
            stemmed_tokens.append(stemmer.stem(token))
        p = re.compile('[a-zA-Z]+')
        pattern_checked = []
        for stem in stemmed_tokens:
            result = stem.replace('.', '')
            result = result.replace(',', '')
            result = result.replace(chr(39), '')
            result = result.replace(chr(34), '')

            if p.match(result) and len(result) >= min_lenght:
                pattern_checked.append(result)

        return pattern_checked

    #pobranie tablicy unikalnych słów dokumentu
    def get_unique_words(self):
        word_list = []
        for word in self.preprocessed_text:
            if not word in word_list:
                word_list.append(word)
        return word_list

    #utworzenie oraz pobranie wektora dokumentu
    def get_vector(self, inverse_vocabulary):
        vector = [0 for i in range(vocabulary_lenght)]
        iterator = 0
        preprocessed_text = self.preprocessed_text
        #na podstawie utworzonego słownika wypełniamy wektor dokumentu, jeżeli dane słowo występuje ustawiamy 1
        for i in inverse_vocabulary.keys():
            if i in preprocessed_text:
                vector[iterator] = 1
            iterator += 1
        return vector


class tf_idf:
    def __init__(self):
        self.D = 0.0
        self.df = {}

    def add_document(self, document):
        self.D += 1.0
        for token in document.get_unique_words():
            if token not in self.df.keys():
                self.df[token] = 1.0
            else:
                self.df[token] += 1.0

    def idf(self, token):
        return math.log(self.D / self.df[token])

    def tf(self, token, document):
        liczba_wystapien_tokenu = 0.0
        liczba_tokenow = 0.0
        for t in document.preprocessed_text:
            liczba_tokenow += 1.0
            if t == token:
                liczba_wystapien_tokenu += 1.0
        return liczba_wystapien_tokenu / liczba_tokenow

    def tfidf(self, token, document):
        return self.tf(token, document) * self.idf(token)


print("Creating corpus...")
crp = corpus()
print("Corpus created!")

print("Creating train vectors")
(X, y) = crp.get_svm_vectors(Train=1)
(X_tfidf, y_tfidf) = crp.get_svm_vectors(Train=1, TfIdf=True)
print("Train vectors created!")

print("Creating test vectors")
(XT, yt) = crp.get_svm_vectors(Test=1)
(XT_tfidf, yt_tfidf) = crp.get_svm_vectors(Test=1, TfIdf=True)
print("Test vectors created!")

print("Starting fitting procedure...")
print("")

klasyfikator = svm.SVC(kernel="linear")
klasyfikator_tfidf = svm.SVC(kernel="linear")
klasyfikator.fit(X, y)
klasyfikator_tfidf.fit(X_tfidf, y_tfidf)

pozytywneSVM = 0
wszystkieSVM = 0

for i, x in enumerate(XT):
    wszystkieSVM += 1
    klasa = klasyfikator.predict(x)
    if klasa == yt[i]:
        pozytywneSVM += 1

pozytywneSVM_tfidf = 0
wszystkieSVM_tfidf = 0

for i, x in enumerate(XT_tfidf):
    wszystkieSVM_tfidf += 1
    klasa = klasyfikator_tfidf.predict(x)
    if klasa == yt_tfidf[i]:
        pozytywneSVM_tfidf += 1

model = GaussianNB()
model.fit(X, y)

pozytywne = 0
wszystkie = 0

for i, x in enumerate(XT):
    wszystkie += 1
    predict = model.predict(x)
    if predict == yt[i]:
        pozytywne += 1

print("SVM:")
print("Pozytywne: {}".format(pozytywneSVM))
print("Wszystkie: {}".format(wszystkieSVM))
print("{}".format(pozytywneSVM / wszystkieSVM))
print("")

print("SVM z TfIdf:")
print("Pozytywne: {}".format(pozytywneSVM_tfidf))
print("Wszystkie: {}".format(wszystkieSVM_tfidf))
print("{}".format(pozytywneSVM_tfidf / wszystkieSVM_tfidf))
print("")

print("NAIVE BAYES:")
print("Pozytywne: {}".format(pozytywne))
print("Wszystkie: {}".format(wszystkie))
print("{}".format(pozytywne / wszystkie))
print("")
