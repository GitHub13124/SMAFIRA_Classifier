# Import packages
import nltk
import os
import pickle
import numpy as np
import pandas as pd

from gensim import corpora
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Constants
DEFAULT_THRESHOLD = 2

class DataHub:
    def __init__(self, labels_filename, abstracts_folder, **kwargs):
        remove_stop = kwargs.get('remove_stop', True)
        threshold = kwargs.get('threshold', DEFAULT_THRESHOLD)

        self.labels_filename = labels_filename
        self.abstracts_folder = abstracts_folder

        self.abstracts_dict = self._buildAbstractsDict()
        self.labels_dict = self._buildLabelsDict()
        self.pmids = np.array(sorted(list(self.labels_dict.keys())))

        self.token_array_dict = self.buildTokenArrayDict(remove_stop = remove_stop, threshold = threshold)
        self.token_count_dict = self._buildTokenCountDict()
        self.sorted_tokens = self.sortTokens()

    # Private functions
    # Helper function for _buildAbstractsDict()
    def __buildFileDict(self):
        file_dict = {}
        for file in os.listdir(self.abstracts_folder):
            if file.endswith('.txt'):
                file_dict[file.split(".txt")[0]] = self.abstracts_folder + "/" + str(file)
        return file_dict

    # Builds a dictionary mapping PMIDs to PubMed abstract strings
    def _buildAbstractsDict(self):
        abstracts_dict = {}
        file_dict =  self.__buildFileDict()
        for file_num in file_dict.keys():
            file = open(file_dict[file_num], "r")
            abstracts_dict[file_num] = ""
            for line in file:
                abstracts_dict[file_num] += line
        return abstracts_dict

    # Builds a dictionary mapping PMIDs to the abstract labels, i.e. in_vivo, ex_vivo, etc.
    def _buildLabelsDict(self):
        labels_dict = {}
        file = open(self.labels_filename, "r")
        for line in file:
            pmid = line.split("\t")[0]
            label = line.split("\t")[1].split("\n")[0]
            labels_dict[pmid] = label
        return labels_dict

    # Builds a dictionary mapping PMIDs to arrays of all 1-grams appearing in the corresponding abstract
    # Filter all tokens that appear fewer than "threshold" times in documents
    def buildTokenArrayDict(self, **kwargs):
        remove_stop = kwargs.get('remove_stop', True)
        threshold = kwargs.get('threshold', DEFAULT_THRESHOLD)
        stop_words = set(stopwords.words('english'))

        token_array_dict = {}
        for pmid in self.pmids:
            text = self.abstracts_dict[pmid]
            tokens = []
            for sentence in nltk.sent_tokenize(text):
                for word in nltk.word_tokenize(sentence):
                    if len(word) >= 2 and not (remove_stop and (word.lower() in stop_words)):
                        tokens.append(word.lower())
            token_array_dict[pmid] = tokens

        self.token_array_dict = token_array_dict

        self._buildTokenCountDict()

        token_array_dict = {pmid: [token for token in array if self.token_count_dict[token] >= threshold] for (pmid, array) in self.token_array_dict.items()}
        self.token_array_dict = token_array_dict

        return token_array_dict

    # Builds a dictionary mapping tokens in the abstracts to the frequency of occurence in the corpus
    def _buildTokenCountDict(self):
        token_array_dict = self.token_array_dict
        token_count_dict = {}
        for pmid in token_array_dict.keys():
            for token in token_array_dict[pmid]:
                if token in token_count_dict.keys():
                    token_count_dict[token] += 1
                else:
                    token_count_dict[token] = 1
        self.token_count_dict = token_count_dict

        return token_count_dict

    # Public functions

    # Sort tokens into alphabetical array
    def sortTokens(self):
        texts = [self.token_array_dict[pmid] for pmid in self.pmids]
        dictionary = corpora.Dictionary(texts)
        sorted_tokens = []
        for i in range(len(dictionary.keys())):
            sorted_tokens.append(dictionary[i])
        sorted_tokens = sorted(list(self.token_count_dict.keys()))
        self.sorted_tokens = sorted_tokens
        return sorted_tokens

    # returns a Pandas DataFrame object storing the data
    # optional parameter to factorize labels into numbers (e.g. "in_vivo" -> 1)
    def buildDataFrame(self, **kwargs):
        factorize = kwargs.get('factorize', None)
        ovr = kwargs.get('ovr', None)

        num_docs = len(self.pmids)
        categories = np.array(["text", "category"])
        num_categories = len(categories)

        data = np.empty((num_docs, num_categories), dtype=object)
        for i in range(num_docs):
            #text column
            data[i][0] = self.abstracts_dict[self.pmids[i]]
            #category column
            if (ovr == None):
                data[i][1] = self.labels_dict[self.pmids[i]]
            else:
                if (self.labels_dict[self.pmids[i]] == ovr):
                    data[i][1] = str(ovr)
                else:
                    data[i][1] = "Not " + str(ovr)

        df = pd.DataFrame(data, index = self.pmids, columns = categories)

        if (factorize):
            factor = pd.factorize(df['category'])
            df.category = factor[0]
            definitions = factor[1]
            return (df, definitions)
        else:
            return df

    # filter out classes (i.e. "in_vivo", "ex_vivo", etc.) specified in "filter_arrays"
    def filterClasses(self, filter_array):
        if len(filter_array) == 0: return
        self.pmids = np.array([pmid for pmid in self.pmids if self.labels_dict[pmid] not in filter_array])
        self.token_array_dict = self.buildTokenArrayDict()
        self.token_count_dict = self._buildTokenCountDict()
        self.sorted_tokens = self.sortTokens()

    # Accessor functions

    def getAbstractsDict(self):
        return self.abstracts_dict

    def getLabels(self):
        return self.labels_dict

    def getPMIDs(self):
        return self.pmids

    def getSortedTokens(self):
        return self.sorted_tokens

    def getTokenArrayDict(self):
        return self.token_array_dict

    def getTokenCountDict(self):
        return self.token_count_dict
