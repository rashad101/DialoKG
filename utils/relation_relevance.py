import numpy as np
from gensim.models import fasttext
from numpy.linalg import norm
from unidecode import unidecode
import json


class LaplacianMatrix:

    def __init__(self, vec=None):
        self.emb = vec

    def adjacency_mat(self, e_r_dict):
        '''
        Generate adjacency matrix from e-r dictionary in conversation
        :param e_r_dict:
        :return: adjacency matrix, identity matrix
        '''
        er_vec = self.getER_vec(e_r_dict)
        dimension = len(er_vec)
        adjacenty_matrix = np.zeros((dimension, dimension))

        for i in range(dimension):
            if er_vec[i] in e_r_dict:  # if element of er_vector[i] is an entity
                for j in range(dimension):
                    if er_vec[j] in e_r_dict[er_vec[i]]:
                        adjacenty_matrix[i][j] = 1.0
            else:  # if element of er_vector[j] is a relation
                for j in range(dimension):
                    for k in range(dimension):
                        if er_vec[k] in e_r_dict and er_vec[j] in e_r_dict[er_vec[k]]:
                            adjacenty_matrix[j][k] = 1.0
        return adjacenty_matrix, np.identity(adjacenty_matrix.shape[0])

    def getER_vec(self,er_dict):
        """
        :param entities: list of entities
        :param relations: list of relations
        :return: genperate ER list -> ["brazil","coach","caps",age"]
        """
        er_vector = []
        for k, v in er_dict.items():
            er_vector.append(k)
            er_vector.extend(v)
        return np.array(er_vector)

    def get_degree_matrix(self, adjacency_matrix):
        """
        :param adjacency_matrix:
        :return: return degree matrix -> example [[1. 0. 0. 0.],
                                                  [0. 2. 0. 0.],
                                                  [0. 0. 0. 0.],
                                                  [0. 0. 0. 3.]]   diagonal numbers represents total number of connections
        """
        return np.array(np.diag(np.array(np.sum(adjacency_matrix, axis=0))))

    def feature_vec(self, q, erlist):
        weight_vector = [self.get_similarity(er_elem, q) if er_elem != "" else 0.0 for er_elem in erlist]
        return weight_vector

    def cosine(self, a, b):
        sim = np.dot(a, b) / (norm(a) * norm(b))
        return sim if not np.isnan(sim) else 0.0

    def get_avg_word2vec(self, phrase):
        """
        get Average word vectors for a given phrases
        """
        vec = np.zeros(300).astype(np.float32)
        if phrase=="":
            return vec
        phrase = unidecode(phrase).strip()
        for w in phrase.split():
            vec = vec + np.array(self.emb.get_vector(w)).reshape(300).astype(np.float32)
        return vec / float(len(phrase.split()))

    def get_similarity(self, er_elem, query):
        """
        :param er_elem: an entity/relation
        :param query: question from an utterance
        :return: cosine similarity value between query and entity/relation
        """
        er_elem_emb = self.get_avg_word2vec(er_elem)
        query_emb = self.get_avg_word2vec(query)
        score = self.cosine(er_elem_emb, query_emb)
        return score

    def relation_relevance(self, q, erdict):
        erlist = self.getER_vec(erdict)
        A,I = self.adjacency_mat(erdict)
        A_hat = A + I
        D = self.get_degree_matrix(A)
        #print(D)
        D_inv = np.linalg.pinv(D)
        dt = np.matmul(D_inv, A_hat)
        X = self.feature_vec(q,erlist)
        h = np.matmul(dt, X)
        current_idx = 0
        mapped_score = dict()
        normalized_scores = self.normalize(h)
        for k,v in erdict.items():
            mapped_score[k] = {
                "original score": h[current_idx],
                "normalized score": normalized_scores[current_idx]
            }
            current_idx+=1
            for val in v:
                mapped_score[val] = {
                    "original score": h[current_idx],
                    "normalized score": normalized_scores[current_idx]
                }
                current_idx += 1

        return h, mapped_score

    def normalize(self, vec):
        mx = np.max(vec)
        norm_vec = np.array([v/(mx+1e-30) for v in vec])
        return norm_vec


if __name__=="__main__":

    vec = fasttext.load_facebook_vectors("data/wiki.simple.bin")
    d = {'manchester united': ['ground',
                                'chairman',
                                'coach',
                                'has player',
                                'jersey color',
                                'founded on'],
         '13 \\( first in 1930 \\)': [],
         'gaston pereiro': ['midfielder',
                                'playes for',
                                'position',
                                'goals',
                                'caps',
                                'age',
                                'jersey']
         }

    gl = LaplacianMatrix(vec=vec)
    question = "Who is the coach of manchester united ?"
    # erlist = gl.getER_vec(d)
    # f_in = gl.feature_vec(question, erlist)
    # print(erlist)
    # print(f_in)
    # hvec = gl.relation_relevance(question,d)
    # print(hvec)
    # normalized_vec = gl.normalize(hvec)
    # print(normalized_vec)
    # gl.rela
    hvec, mapping = gl.relation_relevance(question,d)
    print("hvec: ", hvec)
    print("mapping: ", json.dumps(mapping, indent=2))






