# Import Packages
import os
import pickle
import numpy as np
import pprint as pp

from classes.DataHub import DataHub
from classes.MeshAnalyzer import MeshAnalyzer
from gensim import corpora, models
from sklearn.preprocessing import normalize

# Constants
RANDOM_STATE = 0

class TopicalMeshAnalyzer(MeshAnalyzer):

    def __init__ (self, pmids, mesh_dict_filename, **kwargs):
        super().__init__(pmids, mesh_dict_filename)
        self.mesh_word_matrix_filename = kwargs.get('mesh_word_matrix_filename', "data/mesh_word_matrix")
        self.topic_word_matrix = None
        self.document_topic_matrix = None
        self.mesh_word_matrix = None
        self.topic_mesh_matrix = None
        self.model = None

    # Private Functions

    # Builds the correspondence matrix as described in TopicalMeSH paper
    def _buildCorrespondenceMatrix(self, data, num_topics):
        mesh_word_matrix = self._buildMeshWordMatrix(data)
        topic_word_matrix, document_topic_matrix = self._buildLDAMatrices(data, num_topics)
        topic_mesh_matrix = np.zeros((len(self.topic_word_matrix), len(mesh_word_matrix)))

        for topic_num, topic_row in enumerate(topic_word_matrix):
            vec_1 = topic_row.tolist()
            for mesh_num, mesh_row in enumerate(mesh_word_matrix):
                vec_2 = mesh_row.tolist()
                topic_mesh_matrix[topic_num][mesh_num] = self._rescaledDotProduct(vec_1, vec_2, topic_num, mesh_num)
                #topic_mesh_matrix[topic_num][mesh_num] = np.dot(vec_1, vec_2)

        self.topic_mesh_matrix = topic_mesh_matrix
        return (document_topic_matrix, topic_mesh_matrix)

    # Builds the topic-word matrix, which shows LDA distribution/simplex over words for each topic
    # Builds the document-topic matrix, which shows the topic mixtures for each document
    def _buildLDAMatrices(self, data, num_topics):
        # Get necessary portions of the data
        num_topics = 20
        token_array_dict = data.getTokenArrayDict()
        pmids = data.getPMIDs()

        # Build BoW representations for the text and construct model
        texts = [token_array_dict[pmid] for pmid in pmids]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha = 'asymmetric',
                                 random_state=RANDOM_STATE, passes = 10)
        # 100 passes gave best performance

        # Build LDA matrices
        topic_word_matrix = np.array(model.get_topics())
        document_topic_matrix = np.zeros((len(corpus), num_topics))
        for doc_num, doc in enumerate(corpus):
            topic_distribution = model.get_document_topics(doc)
            for topic in topic_distribution:
                topic_num = topic[0]
                topic_value = topic[1]
                document_topic_matrix[doc_num][topic_num] = topic_value

        # Store matrices and model in TopicalMeSH object
        self.topic_word_matrix = topic_word_matrix
        self.document_topic_matrix = document_topic_matrix
        self.model = model

        return (topic_word_matrix, document_topic_matrix)

    # Builds a MeSH topic array, which is a tf-idf weighted distribution of words for a particular MeSH term
    def _buildMeshTopic(self, mesh, data):
        if (mesh not in self.mesh_count_dict.keys()):
            print("MeSH term does not exist.")
            return

        mesh_dict = self.mesh_dict
        token_array_dict = data.getTokenArrayDict()
        token_count_dict = data.getTokenCountDict()
        token_idf_dict = self._getMeshTokenIDF(mesh, data)
        sorted_tokens = data.sortTokens()

        # Initialize mesh_topic_dict, which maps words to tf-idf for tokens in the abstract
        # Use 1 to add Laplace smoothing
        mesh_topic_dict = {token: 1 for token in sorted_tokens}
        for token in sorted_tokens:
            for pmid in self.pmids:
                # if word is in the current abstract and this abstract is labeled with MeSH term
                if (token in token_array_dict[pmid]) and (mesh in mesh_dict[pmid]):
                    mesh_topic_dict[token] += token_array_dict[pmid].count(token)

        # Rescale by idf
        mesh_topic_dict = {token : tf * token_idf_dict[token] for token, tf in mesh_topic_dict.items()}
        mesh_topic_array = np.array([mesh_topic_dict[token] for token in sorted_tokens])
        mesh_topic_array /= np.linalg.norm(mesh_topic_array)
        return mesh_topic_array

    # Builds the MeSH-word matrix, which gives a tf-idf weighting of words for each MeSH term, as if they were topics
    def _buildMeshWordMatrix(self, data):
        if(os.path.exists(self.mesh_word_matrix_filename + ".pkl")):
            pickle_in = open(self.mesh_word_matrix_filename + ".pkl","rb")
            mesh_word_matrix = pickle.load(pickle_in)
            self.mesh_word_matrix = mesh_word_matrix
            return mesh_word_matrix

        mesh_word_matrix = []
        for mesh_num, mesh in enumerate(self.sorted_mesh_terms):
            mesh_word_matrix.append(self._buildMeshTopic(mesh, data))
            print("Mesh Number: " + str(mesh_num))
        mesh_word_matrix = np.array(mesh_word_matrix)
        self.mesh_word_matrix = mesh_word_matrix

        # Saves the MeSH-word matrix in a pickle file
        with open(self.mesh_word_matrix_filename + ".pkl", "wb") as f:
            pickle.dump(mesh_word_matrix, f)

        return mesh_word_matrix

    # Returns a dictionary returning the IDF for each token corresponding to a particular MeSH term
    def _getMeshTokenIDF(self, mesh, data):
        num_docs = len(self.pmids)
        sorted_tokens = data.getSortedTokens()
        token_array_dict = data.getTokenArrayDict()
        # start with 1.0 for Laplace smoothing
        token_idf_dict = {token: 1.0 for token in sorted_tokens}
        for token in sorted_tokens:
            for pmid in self.pmids:
                if token in token_array_dict[pmid] and mesh in self.mesh_dict[pmid]:
                    token_idf_dict[token] += 1.0
        for token in token_idf_dict.keys():
            token_idf_dict[token] = np.log(num_docs / (token_idf_dict[token]))

        return token_idf_dict

    # Computes the rescaled dot product, as defined in TopicalMeSH paper
    def _rescaledDotProduct(self, vector_1, vector_2, topic_num, mesh_num):
        dot_product = np.dot(vector_1, vector_2)
        d_max = np.dot(sorted(vector_1), sorted(vector_2))
        d_min = np.dot(sorted(vector_1, reverse = True), sorted(vector_2))
        return (dot_product - d_min) / (d_max - d_min)

    # Public Functions

    # Builds the document-topical-MeSH matrix, which is used as embeddings for downstream classification tasks
    def buildEmbeddingMatrix(self, data, **kwargs):
        num_topics = kwargs.get('num_topics', len(self.mesh_count_dict.keys()))
        document_topic_matrix, topic_mesh_matrix = self._buildCorrespondenceMatrix(data, num_topics)
        document_topical_mesh_matrix = np.dot(document_topic_matrix, topic_mesh_matrix)
        return document_topic_matrix
