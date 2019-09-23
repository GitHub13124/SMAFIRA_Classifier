# Import packages
import numpy as np

# Constants
RANDOM_STATE = 0

class CircularClassifierChain:

    def __init__(self, train, test, classifiers):
        self.train = train
        self.train_pmids = np.array(list(train.index))
        self.test = test
        self.classifiers = classifiers
        self.num_classifiers = len(classifiers)
        self.predictions_dict = {}

    def _run(self, **kwargs):
        num_cycles = kwargs.get('num_cycles', 10)
        # Dictionary storing the "votes" from earlier classifiers in the chain, mapping PMIDs to an array of such "votes"
        attribute_dict = {pmid: np.zeros(self.num_classifiers) for pmid in self.train_pmids}
        predictions_dict = {pmid: np.zeros(self.num_classifiers) for pmid in self.train_pmids}
        for cycle_num in range(num_cycles):
            print("Cycle: " + str(cycle_num))
            for classifier_num, classifier in enumerate(self.classifiers):
                print("Classifier: " + str(classifier_num))
                classifier._trainModel(attribute_dict = attribute_dict)
                y_pred = classifier.testModel(self.test)
                for pmid_num, pmid in enumerate(self.train_pmids):
                    if y_pred[pmid_num] == 0:
                        attribute_dict[pmid][classifier_num] = -1
                    else:
                        attribute_dict[pmid][classifier_num] = 1
                    print(attribute_dict[pmid])
                    predictions_dict[pmid] = attribute_dict[pmid]
                    y_pred[(pmid_num + 1 % self.num_classifiers)] = 0
        self.predictions_dict = predictions_dict
