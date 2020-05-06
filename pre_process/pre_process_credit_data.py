import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class PreProcessCredit:
    __slots__ = ['__base', '__predictors', '__classes']

    __base_name = 'credit_data.csv'

    def __init__(self):
        self.__base = pd.read_csv(self.__base_name)
        self.__predictors()
        self.__classes()
        self.__treatment()

    def __predictors(self):
        self.__predictors = self.__base.iloc[:, 1:4].values

    def __classes(self):
        self.__classes = self.__base.iloc[:, 4].values

    def __treatment(self):
        self.__treatment_negative_age()
        self.__treatment_null_values()
        self.__scaling()

    def __treatment_negative_age(self):
        mean = self.__base['age'][self.__base.age > 0].mean()
        self.__base.loc[self.__base.age < 0, 'age'] = mean

    def __treatment_null_values(self):
        imputer = SimpleImputer().fit(self.__predictors[:, 0:3])
        self.__predictors[:, 0:3] = imputer.transform(self.__predictors[:, 0:3])

    def __scaling(self):
        scaler = StandardScaler()
        self.__predictors = scaler.fit_transform(self.__predictors)

    @property
    def base(self):
        return self.__base

    @property
    def predictors(self):
        return self.__predictors

    @property
    def classes(self):
        return self.__classes
