from pandas import read_csv
from numpy import nan
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class PreProcessCredit:
    __slots__ = ['__base', '__predictors', '__classes', '__imputer']

    __base_name = 'credit_data.csv'
    __scaler = StandardScaler()

    def __init__(self):
        self.__base = read_csv(self.__base_name)
        self.__imputer = SimpleImputer(strategy='mean', missing_values=nan)
        self.__set_predictors()
        self.__set_classes()
        self.__treatment()

    def __set_predictors(self):
        self.__predictors = self.__base.iloc[:, 1:4].values

    def __set_classes(self):
        self.__classes = self.__base.iloc[:, 4].values

    def __treatment(self):
        self.__treatment_negative_age()
        self.__treatment_null_values()
        self.__scaling()

    def __treatment_negative_age(self):
        mean = self.__base['age'][self.__base.age > 0].mean()
        self.__base.loc[self.__base.age < 0, 'age'] = mean

    def __treatment_null_values(self):
        self.__imputer = self.__imputer.fit(self.__predictors[:, 1:4])
        self.__predictors[:, 1:4] = self.__imputer.transform(self.__predictors[:, 1:4])

    def __scaling(self):
        self.__predictors = self.__scaler.fit_transform(self.__predictors)

    @property
    def predictors(self):
        return self.__predictors

    @property
    def classes(self):
        return self.__classes
