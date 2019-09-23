# Import packages
import numpy as np

from classes.ClassifierBlock import ClassifierBlock

# Constants
RANDOM_STATE = 0

class ClassifierNode(ClassifierBlock):
    def __init__(self, train, embedding_dict, classifier_type, **kwargs)
        num_estimators_ABC = kwargs.get('num_estimators_ABC', 250)
        imbalance_type = kwargs.get('imbalance_type', None)
        super().__init__(train, embedding_dict, classifier_type, num_estimators_ABC = num_estimators_ABC, imbalance_type = imbalance_type)

        self.children_nodes = []
        self.is_leaf = True
        #self.predictions = []

    def addChildNodes(self, child_node_array):
        for child_node in child_node_array:
            self.children_nodes.append(child_node)
        if (len(self.children_nodes) > 0): self.is_leaf = False

    def test(self):
        if (is_leaf): return
