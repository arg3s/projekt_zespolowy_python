from scipy.spatial.distance import cosine
import math

fs = open("C:\\Users\\s0152868\\Desktop\\glove.6B.100d.txt", 'r', encoding='utf8')
model = {}


def cosine2(v1, v2):
    v1v2 = 0
    v1norm = 0
    v2norm = 0
    for i in range(100):
        v1v2 += v1[i] * v2[i]
        v1norm += v1[i] * v1[i]
        v2norm += v2[i] * v2[i]
    v1norm = math.sqrt(v1norm)
    v2norm = math.sqrt(v2norm)
    return v1v2 / (v1norm * v2norm)


for line in fs:
    tokens = line.split()
    vec = [0 for i in range(100)]
    for i in range(100):
        vec[i] = float(tokens[i + 1])
        model[tokens[0]] = vec

king_vec = model["king"]
queen_vec = model["queen"]
man_vec = model["man"]
woman_vec = model["woman"]

print(cosine2(king_vec, man_vec))
print(cosine2(queen_vec, woman_vec))
print(cosine2(king_vec, queen_vec))
print(cosine2(king_vec, woman_vec))

print(cosine(king_vec, man_vec))
print(cosine(queen_vec, woman_vec))
print(cosine(king_vec, queen_vec))
print(cosine(king_vec, woman_vec))
