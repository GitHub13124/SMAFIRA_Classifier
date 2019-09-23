# Import packages

from classes.CircularClassifierChain import CircularClassifierChain
from classes.ClassifierBlock import ClassifierBlock
from classes.ClassifierChain import ClassifierChain
from classes.DataHub import DataHub
from classes.Embedder import Embedder
from sklearn.model_selection import train_test_split

# ?
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
import pickle


# Constants
RANDOM_STATE = 15

def main(labels_filename, abstracts_folder):
    data = DataHub(labels_filename, abstracts_folder)
    pmids = data.getPMIDs()

    embedder = Embedder(data)
    embedding_dict = embedder.getMergedEmbedding(embedder.getBertEmbedding(), embedder.getMeSHEmbedding())
    embedding_dict = embedder.getW2VEmbedding()

    data.filterClasses(["in_vivo_invertebrates"])
    df, definitions = data.buildDataFrame(factorize = True, ovr = "in_vivo")

    train, dev = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE)

    svm_block = ClassifierBlock(train, embedding_dict, "SVC", imbalance_type = "ros", normalize = True)
    svm_block.testModel(dev)


main("data/labels.txt", "data/abstracts")
