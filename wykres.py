from matplotlib import pyplot as plt
from sklearn import svm

def load_data(filename):
    matrix = []
    for line in open(filename):
        tokens = line.split()
        matrix.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])
    return matrix


def draw_data(matrix):
    xs1 = []
    ys1 = []
    xs0 = []
    ys0 = []
    xsn = []
    ysn = []
    for[x, y, c] in matrix:
        if c == 1:
            xs1.append(x)
            ys1.append(y)
        if c == 0:
            xs0.append(x)
            ys0.append(y)
        if c == -1:
            xsn.append(x)
            ysn.append(y)
    plt.axis([-5, 15, -5, 15])
    plt.plot(xs0, ys0, 'ro', color='red')
    plt.plot(xs1, ys1, 'ro', color='blue')
    plt.plot(xsn, ysn, 'ro', color='yellow')
    plt.show()
    

def classify(matrix):
    xs = []
    ys = []
    for[x, y, c] in matrix:
        if c == 1 or c == 0:
            xs.append([x, y])
            ys.append(c)
    classifier = svm.SVC()
    classifier.fit(xs, ys)
    for i in range(len(matrix)):
        if matrix[i][2] == -1:
            matrix[i][2] = classifier.predict([matrix[i][0], matrix[i][1]])
    return matrix
matrix = load_data('train')
draw_data(matrix)
draw_data(classify(matrix))
