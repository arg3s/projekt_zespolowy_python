from os import listdir
from plik import tf_idf
from plik import feature_values
from sklearn import svm

class Corpus:
    def __init__(self, dir_pos, dir_neg):
        self.documents = []
        self.dir_pos = dir_pos
        self.dir_neg = dir_neg
        for i, file in enumerate(listdir(dir_neg)):
            # print(i)
            if i < 300:
                fs = open(dir_neg + "\\" + file, 'r')
                text = fs.read()
                positive = 0
                train = 0
                doc = Document(text, positive, train)
                self.add_document(doc)
            else:
                fs = open(dir_neg + "\\" + file, 'r')
                text = fs.read()
                positive = 0
                train = 1
                doc = Document(text, positive, train)
                self.add_document(doc)

        for i, file in enumerate(listdir(dir_pos)):
            # print(i)
            if i < 300:
                fs = open(dir_pos + "\\" + file, 'r')
                text = fs.read()
                positive = 1
                train = 0
                doc = Document(text, positive, train)
                self.add_document(doc)
            else:
                fs = open(dir_pos + "\\" + file, 'r')
                text = fs.read()
                positive = 1
                train = 1
                doc = Document(text, positive, train)
                self.add_document(doc)
                # posdocs.append(open(dir_pos + "\\" + file, 'r').read())

    def add_document(self, document):
        self.documents.append(document)

    def get_train_documents(self):
        train = []
        for doc in self.documents:
            if doc.train == 1:
                train.append(doc.text)
        return train

#    def get_representer(self):
#       return tf_idf(self.get_train_documents())

    def initialize_vocabulary(self):
        self.vocabulary = {}
        self.inverse_vocabulary = {}
        for i, doc in enumerate(self.documents):
            if i%1000 == 0:
                print(i)
            for word in doc.get_unique_words():
                if word not in self.vocabulary:
                    self.vocabulary[i] = word
                    self.inverse_vocabulary[word] = i

    def get_svm_vectors(self, train = 0, test = 0):
        Xs = []
        Ys = []
        for doc in self.documents:
            if train == 1 and doc.train == 0:
                continue
            if test == 1 and doc.train == 1:
                continue
            x = doc.get_vector(self.inverse_vocabulary)
            y = doc.positive
            Xs.append(x)
            Ys.append(y)
        return (Xs, Ys)

class Document:
    def __init__(self, text, positive=1, train=1):
        self.positive = positive
        self.train = train
        self.text = text

    def get_feature_values(self, representer):
        return feature_values(self.text, representer)

    def get_unique_words(self):
        word_list = []
        for word in self.text.split():
            if not word in word_list:
                word_list.append(word)
        return word_list

    def get_vector(self, inverse_vocabulary):
        lng = len(inverse_vocabulary)
        vector = [0 for i in range(lng)]
        for word in self.text.split():
            vector[inverse_vocabulary[word]] = 1
        return vector



crp = Corpus("C:\\Users\\s0152868\\Desktop\\txt_sentoken\\pos", "C:\\Users\\s0152868\\Desktop\\txt_sentoken\\neg")
crp.get_train_documents()
crp.initialize_vocabulary()
print(crp.vocabulary)

klasyfikator = svm.SVC(kernel="linear")  # svm.SVC(kernel = "linear")
(X, y) = crp.get_svm_vectors(train=1)

print("starting fitting procedure")
klasyfikator.fit(X, y)

(XT, yt) = crp.get_svm_vectors(test=1)

pozytywne = 0
wszystkie = 0
for i, x in enumerate(XT):
    wszystkie += 1
    klasa = klasyfikator.predict(x)
    if klasa == yt[i]:
        pozytywne += 1

print(pozytywne)
print(wszystkie)