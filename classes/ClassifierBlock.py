# Import packages
import pandas
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

# Constants
RANDOM_STATE = 0

class ClassifierBlock:
    def __init__(self, train, embedding_dict, classifier_type, **kwargs):
        num_estimators_ABC = kwargs.get('num_estimators_ABC', 250)
        self.imbalance_type = kwargs.get('imbalance_type', None)
        self.normalize = kwargs.get('normalize', False)

        self.train = train
        self.embedding_dict = embedding_dict
        self.classifier_type = classifier_type

        self.model = self._trainModel(num_estimators_ABC = num_estimators_ABC)

    # Private functions

    # Build Y
    def _buildY(self, data_frame):
        # Build y_train
        y = []
        for pmid, row in data_frame.iterrows():
            y.append(row['category'])
        self.y_train = np.array(y)
        return self.y_train

    # Build X for a given embedding
    def _buildX(self, data_frame, **kwargs):
        attribute_dict = kwargs.get('attribute_dict', None)
        X = []
        for pmid, row in data_frame.iterrows():
            feature_vec = self.embedding_dict[pmid]
            if (attribute_dict != None):
                feature_vec = np.concatenate([feature_vec, attribute_dict[pmid]])
            X.append(feature_vec)
        X = np.array(X)
        if (self.normalize): X = normalize(X)
        return X

    def _trainModel(self, **kwargs):
        num_estimators_ABC = kwargs.get('num_estimators_ABC', 250)
        attribute_dict = kwargs.get('attribute_dict', None)

        if (self.classifier_type not in ['ABC', 'LR', 'NB', 'RFC', 'SVC']):
            print("Error: Not a valid machine-learning algorithm.")
            return
            
        y_train = self._buildY(self.train)
        X_train = self._buildX(self.train, attribute_dict = attribute_dict)

        # Address class imbalance
        if (self.imbalance_type == 'ros'):
            X_train, y_train = RandomOverSampler(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
        elif (self.imbalance_type == 'smote'):
            X_train, y_train, = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)

        if (self.classifier_type == 'ABC'):
            model = AdaBoostClassifier(n_estimators = num_estimators_ABC, random_state = RANDOM_STATE).fit(X_train, y_train)
        elif (self.classifier_type == 'LR'):
            model = LogisticRegression(C=1e5, random_state = RANDOM_STATE).fit(X_train, y_train)
        elif (self.classifier_type == 'NB'):
            model = MultinomialNB(random_state = RANDOM_STATE).fit(X_train, y_train)
        elif (self.classifier_type == 'RFC'):
            model = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = RANDOM_STATE).fit(X_train, y_train)
        elif (self.classifier_type == 'SVC'):
            model = SVC(kernel = 'linear', C = 1, random_state = RANDOM_STATE).fit(X_train, y_train)

        self.model = model

        return model

    # Private functions
    def testModel(self, test):
        y_test = self._buildY(test)
        X_test = self._buildX(test)

        y_pred = self.model.predict(X_test)

        print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
        print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

        # creating a confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        return y_pred
