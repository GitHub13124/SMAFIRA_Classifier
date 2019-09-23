# Import packages
import json
import nltk
import os
import pickle
import sent2vec
import gensim.models.keyedvectors as word2vec
import numpy as np

from allennlp.commands.elmo import ElmoEmbedder
from classes.TopicalMeshAnalyzer import TopicalMeshAnalyzer
from classes.MeshAnalyzer import MeshAnalyzer
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation

# Constants
NUM_BERT_FEATURES = 768
NUM_SENT_VEC_FEATURES = 700
NUM_W2V_FEATURES = 200

class Embedder:
    def __init__(self, data, **kwargs):
        self.bert_embedding_filename = kwargs.get('bert_filename', "data/embeddings/bert_embedding")
        self.bow_embedding_filename = kwargs.get('bow_filename', "data/embeddings/bow_embedding")
        self.elmo_embedding_filename = kwargs.get('elmo_filename', "data/embeddings/elmo_embedding")
        self.one_hot_embedding_filename = kwargs.get('one_hot_filename', "data/embeddings/one_hot_embedding")
        self.sent_vec_embedding_filename = kwargs.get('sent_vec_filename', "data/embeddings/sent_vec_embedding")
        self.w2v_embedding_filename = kwargs.get('w2v_filename', "data/embeddings/w2v_embedding")

        self.mesh_embedding_filename = kwargs.get('mesh_filename', "data/embeddings/mesh_embedding")
        self.topical_mesh_embedding_filename = kwargs.get('topical_mesh_filename', "data/embeddings/topical_mesh_embedding")

        self.data = data
        self.pmids = data.getPMIDs()
        self.topical_mesh_analyzer = None
        self.mesh_analyzer = None

        self.bert_embedding = None
        self.bow_embedding = None
        self.elmo_embedding = None
        self.one_hot_embedding = None
        self.sent_vec_embedding = None
        self.w2v_embedding = None

        self.mesh_embedding = None
        self.topical_mesh_embedding = None


    # Build Abstract Embeddings

    # Builds a BioBERT embedding given the desired pre-training
    def _buildBertEmbedding(self, **kwargs):
        pretrain_name = kwargs.get('pretrain_name', "biobert_v1.1_pubmed")

        if (pretrain_name not in ["biobert_v1.0_pubmed", "biobert_v1.0_pubmed_pmc", "biobert_v1.1_pubmed"]):
            print("Error: Not a valid pre-trained model.")
            return

        if(os.path.exists(self.bert_embedding_filename + ".pkl")):
            pickle_in = open(self.bert_embedding_filename + ".pkl","rb")
            bert_embedding = pickle.load(pickle_in)
            self.bert_embedding = bert_embedding
            return bert_embedding

        abstracts_dict = self.data.getAbstractsDict()

        #input_file = open("data/bert/input.txt", "w")
        #for pmid in self.pmids:
        #    input_file.write(abstracts_dict[pmid])
        #    input_file.write("\n")
        #input_file.close()

        if (pretrain_name in ["biobert_v1.0_pubmed", "biobert_v1.0_pubmed_pmc"]):
            os.system('python extract_bert_features.py \
                --input_file=./data/bert/input.txt \
                --output_file=./data/bert/' + pretrain_name + '/output.jsonl \
                --vocab_file=./data/bert/' + pretrain_name + '/vocab.txt \
                --bert_config_file=./data/bert/' + pretrain_name + '/bert_config.json \
                --init_checkpoint=./data/bert/' + pretrain_name + '/biobert_model.ckpt \
                --layers=-1,-2,-3,-4 \
                --max_seq_length=450')
        else:
            os.system('python extract_bert_features.py \
                --input_file=./data/bert/input.txt \
                --output_file=./data/bert/' + pretrain_name + '/output.jsonl \
                --vocab_file=./data/bert/' + pretrain_name + '/vocab.txt \
                --bert_config_file=./data/bert/' + pretrain_name + '/bert_config.json \
                --init_checkpoint=./data/bert/' + pretrain_name + '/model.ckpt-1000000 \
                --layers=-1,-2,-3,-4 \
                --max_seq_length=450')

        bert_embedding = {}
        with open('data/bert/' + pretrain_name + '/output.jsonl') as f:
            count = 0
            for line in f:
                if (line == ""): print("BLANK")
                if count == 825: continue
                bert_layers = json.loads(line)
                print(count)

                document_vector = np.zeros(2 * NUM_BERT_FEATURES)
                #document_vector = np.zeros(NUM_BERT_FEATURES)

                # Counts the number of [SEP] tokens in addition to the final period
                num_sep = 0

                # Starts from 1 to skip [CLS] and to -1 to skip [SEP]
                for i in range (1, len(bert_layers['features'])-1):
                    if bert_layers['features'][i]['token'] == ['[SEP]']:
                        num_sep += 1
                        continue
                    # Feature vector of the current token
                    #token_vector = np.array(bert_layers['features'][i]['layers'][0]['values'])
                    token_vector = np.concatenate([np.array(bert_layers['features'][i]['layers'][0]['values']),
                                                   np.array(bert_layers['features'][i]['layers'][1]['values'])])
                    # Adding token vector to the document vector
                    document_vector = document_vector + token_vector

                # -2 is for excluding [CLS] and [SEP] tokens
                number_of_tokens = len(bert_layers['features']) - 2 - num_sep

                # Since we want to compute the average of vector representations of tokens
                if (number_of_tokens != 0):
                    document_vector = np.divide(document_vector, number_of_tokens)

                bert_embedding[self.pmids[count]] = document_vector
                count += 1

        # Saving the embedding dictionary in a pickle file
        with open(self.bert_embedding_filename + ".pkl", "wb") as f:
            pickle.dump(bert_embedding, f)

        self.bert_embedding = bert_embedding

        return bert_embedding


    # Builds an embedding using BioSentVec
    def _buildBioSentVecEmbedding(self):
        if(os.path.exists(self.sent_vec_embedding_filename + ".pkl")):
            pickle_in = open(self.sent_vec_embedding_filename + ".pkl","rb")
            sent_vec_embedding = pickle.load(pickle_in)
            self.sent_vec_embedding = sent_vec_embedding
            return sent_vec_embedding

        model_path = "BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
        model = sent2vec.Sent2vecModel()
        model.load_model(model_path)

        abstracts_dict = self.data.getAbstractsDict()

        sent_vec_embedding = {}

        for pmid in self.pmids:
            text = abstracts_dict[pmid]
            document_vector = np.zeros((NUM_SENT_VEC_FEATURES,))
            for sentence in nltk.sent_tokenize(text):
                processed_sentence = self.__sentVecPreprocessSentence(sentence)
                document_vector += model.embed_sentence(processed_sentence).reshape((NUM_SENT_VEC_FEATURES,))
            sent_vec_embedding[pmid] = document_vector

        # Saving the embedding dictionary in a pickle file
        with open(self.sent_vec_embedding_filename + ".pkl", "wb") as f:
            pickle.dump(sent_vec_embedding, f)

        self.sent_vec_embedding = sent_vec_embedding


        return sent_vec_embedding

    # Builds a Bag-of-Words embedding
    def _buildBoWEmbedding(self):
        if(os.path.exists(self.bow_embedding_filename + ".pkl")):
            pickle_in = open(self.bow_embedding_filename + ".pkl","rb")
            bow_embedding = pickle.load(pickle_in)
            self.bow_embedding = bow_embedding
            return bow_embedding

        token_array_dict = self.data.getTokenArrayDict()

        # Build BoW representations for the text and construct model
        texts = [token_array_dict[pmid] for pmid in self.pmids]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        bow_embedding = {}
        for pmid_num, text in enumerate(corpus):
            embedding_vector = np.zeros(len(dictionary))
            for pair in text:
                embedding_vector[pair[0]] = pair[1]
            bow_embedding[self.pmids[pmid_num]] = embedding_vector

        # Saving the embedding dictionary in a pickle file
        with open(self.bow_embedding_filename + ".pkl", "wb") as f:
            pickle.dump(bow_embedding, f)

        self.bow_embedding = bow_embedding

        return bow_embedding

    # Builds an embedding using BioELMo
    def _buildElmoEmbedding(self, **kwargs):
        if(os.path.exists(self.elmo_embedding_filename + ".pkl")):
            pickle_in = open(self.elmo_embedding_filename + ".pkl","rb")
            elmo_embedding = pickle.load(pickle_in)
            self.elmo_embedding = elmo_embedding
            return elmo_embedding

        layer_num = kwargs.get('layer_num', 2)

        vocab = np.genfromtxt('data/elmo/vocabulary.txt', dtype='str')
        options_file = "data/elmo/biomed_elmo_options.json"
        weight_file = "data/elmo/biomed_elmo_weights.hdf5"

        elmo = ElmoEmbedder(options_file, weight_file)

        token_array_dict = self.data.getTokenArrayDict()

        elmo_embedding = {}
        for pmid in self.pmids:
            tokens = token_array_dict[pmid]
            # Embedding is of shape (# of layers, # of tokens, 1024)
            # We take the top layer (the second layer or whichever was provided) and then sum over all tokens
            elmo_embedding[pmid] = np.sum(elmo.embed_sentence(tokens)[layer_num], axis = 0)

        # Saving the embedding dictionary in a pickle file
        with open(self.elmo_embedding_filename + ".pkl", "wb") as f:
            pickle.dump(elmo_embedding, f)

        self.elmo_embedding = elmo_embedding

        return elmo_embedding

    # Builds a Bag-of-Words embedding with one-hot encoding
    def _buildOneHotEmbedding(self):
        if(os.path.exists(self.one_hot_embedding_filename + ".pkl")):
            pickle_in = open(self.one_hot_embedding_filename + ".pkl","rb")
            one_hot_embedding = pickle.load(pickle_in)
            self.one_hot_embedding = one_hot_embedding
            return one_hot_embedding

        one_hot_embedding = {}
        bow_embedding = self._buildBoWEmbedding()
        for pmid in bow_embedding.keys():
            one_hot_embedding[pmid] = np.array([int(bow_embedding[pmid][word_num] > 0) for word_num in range(len(bow_embedding[pmid]))])

        # Saving the embedding dictionary in a pickle file
        with open(self.one_hot_embedding_filename + ".pkl", "wb") as f:
            pickle.dump(one_hot_embedding, f)

        self.one_hot_embedding = one_hot_embedding

        return one_hot_embedding


    # Builds a PubMed-trained word2vec embedding
    def _buildW2VEmbedding(self):
        if(os.path.exists(self.w2v_embedding_filename + ".pkl")):
            pickle_in = open(self.w2v_embedding_filename + ".pkl","rb")
            w2v_embedding = pickle.load(pickle_in)
            self.w2v_embedding = w2v_embedding
            return w2v_embedding

        model = word2vec.KeyedVectors.load_word2vec_format('data/w2v/PubMed-w2v.bin', binary=True)

        token_array_dict = self.data.getTokenArrayDict()

        w2v_embedding = {}
        for pmid in self.pmids:
            tokens = token_array_dict[pmid]
            document_vector = np.zeros((NUM_W2V_FEATURES,))
            for token in tokens:
                if token in model.vocab: document_vector = document_vector + model[token]
            document_vector = np.divide(document_vector, np.linalg.norm(document_vector))
            w2v_embedding[pmid] = document_vector

        # Saving the embedding dictionary in a pickle file
        with open(self.w2v_embedding_filename + ".pkl", "wb") as f:
            pickle.dump(w2v_embedding, f)

        self.w2v_embedding = w2v_embedding

        return w2v_embedding

    # Helper function for BioSentVec
    def __sentVecPreprocessSentence(self, text):
        stop_words = set(stopwords.words('english'))
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()

        tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

        return ' '.join(tokens)

    # Build MeSH Embeddings

    # Builds an embedding using TopicalMeSH
    # The "threshold" attribute filters out all MeSH terms paired with fewer than "threshold" documents
    def _buildTopicalMeSHEmbedding(self, **kwargs):
        if(os.path.exists(self.topical_mesh_embedding_filename + ".pkl")):
            pickle_in = open(self.topical_mesh_embedding_filename + ".pkl","rb")
            topical_mesh_embedding = pickle.load(pickle_in)
            self.topical_mesh_embedding = topical_mesh_embedding
            return topical_mesh_embedding

        threshold = kwargs.get("threshold", 3)

        if (self.topical_mesh_analyzer == None):
            mesh_analyzer = TopicalMeshAnalyzer(self.pmids, "data/mesh_dict")
            self.topical_mesh_analyzer = mesh_analyzer

        mesh_analyzer.filterMesh(3)

        document_topical_mesh_matrix = mesh_analyzer.buildEmbeddingMatrix(self.data)
        topical_mesh_embedding = {}
        for doc_num, pmid in enumerate(self.pmids):
            topical_mesh_embedding[pmid] = document_topical_mesh_matrix[doc_num]

        self.topical_mesh_embedding = topical_mesh_embedding

        return topical_mesh_embedding

    # Builds a one-hot embedding of MeSH terms
    # The "threshold" attribute filters out all MeSH terms paired with fewer than "threshold" documents
    def _buildMeSHEmbedding(self, **kwargs):
        if(os.path.exists(self.mesh_embedding_filename + ".pkl")):
            pickle_in = open(self.mesh_embedding_filename + ".pkl","rb")
            mesh_embedding = pickle.load(pickle_in)
            self.mesh_embedding = mesh_embedding
            return mesh_embedding

        threshold = kwargs.get('threshold', 3)

        if (self.mesh_analyzer == None):
            mesh_analyzer = MeshAnalyzer(self.pmids, "data/mesh_dict")
            self.mesh_analyzer = mesh_analyzer

        mesh_analyzer.filterMesh(threshold)

        document_mesh_matrix = mesh_analyzer.getDocumentMeshMatrix()
        mesh_embedding = {}
        for doc_num, pmid in enumerate(self.pmids):
            mesh_embedding[pmid] = document_mesh_matrix[doc_num]

        # Saving the embedding dictionary in a pickle file
        with open(self.mesh_embedding_filename + ".pkl", "wb") as f:
            pickle.dump(mesh_embedding, f)

        self.mesh_embedding = mesh_embedding

        return mesh_embedding

    # Accessors

    # Returns a concatenated version of two embeddings
    def getMergedEmbedding(self, embedding_1, embedding_2):
        merged_embedding = {}
        for pmid in self.pmids:
            merged_embedding[pmid] = np.concatenate([embedding_1[pmid], embedding_2[pmid]])
        return merged_embedding

    # Abstract Embedding Accessors

    def getBertEmbedding(self):
        if (self.bert_embedding == None): self._buildBertEmbedding()
        return self.bert_embedding

    def getBioSentVecEmbedding(self):
        if (self.sent_vec_embedding == None): self._buildBioSentVecEmbedding()
        return self.sent_vec_embedding

    def getBoWEmbedding(self):
        if (self.bow_embedding == None): self._buildBoWEmbedding()
        return self.bow_embedding

    def getElmoEmbedding(self):
        if (self.elmo_embedding == None): self._buildElmoEmbedding()
        return self.elmo_embedding

    def getOneHotEmbedding(self):
        if (self.one_hot_embedding == None): self._buildOneHotEmbedding()
        return self.one_hot_embedding

    def getW2VEmbedding(self):
        if (self.w2v_embedding == None): self._buildW2VEmbedding()
        return self.w2v_embedding

    # MeSH Embedding Accessors

    def getMeSHEmbedding(self):
        if (self.mesh_embedding == None): self._buildMeSHEmbedding()
        return self.mesh_embedding

    def getTopicalMeSHEmbedding(self):
        if (self.topical_mesh_embedding == None): self._buildTopicalMeSHEmbedding()
        return self.topical_mesh_embedding
