from enum import Enum
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


class Tree:
    __slots__ = ['__classifier', '__f_providers', '__f_classes', '__t_providers', '__t_classes', '__base',
                 '__precision', '__matrix']

    def __init__(self, base, criterion):
        self.__base = base
        self.__classifier = DecisionTreeClassifier(criterion=criterion, random_state=0)
        self.__f_providers, self.__f_classes = self.__base.training_base()
        self.__t_providers, self.__t_classes = self.__base.test_base()
        self.__fit_base()

    def __fit_base(self):
        self.__classifier.fit(self.__f_providers, self.__f_classes)

    def _test_base(self):
        self.__check_results(self.__predicts())

    def __check_results(self, predicts):
        self.__precision = accuracy_score(self.__t_classes, predicts) * 100
        self.__matrix = confusion_matrix(self.__t_classes, predicts)

    def __predicts(self):
        return self.__classifier.predict(self.__t_providers)

    @property
    def precision(self):
        return self.__precision

    @property
    def matrix(self):
        return self.__matrix


class TreeCriterion(Enum):
    ENTROPY = 'entropy'
    GINI = 'gini'
