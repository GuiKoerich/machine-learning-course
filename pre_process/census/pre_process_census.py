from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class PreProcessCensus:
    __slots__ = ['__base', '__predictors', '__classes']

    __csv = '/home/guilherme/PycharmProjects/ia/pre_process/census/census.csv'
    __label_encoder = LabelEncoder()
    __columns_to_encode = [1, 3, 5, 6, 7, 8, 9, 13]
    __predictors_dummy = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), __columns_to_encode)],
                                           remainder='passthrough')
    __scaler = StandardScaler()

    def __init__(self):
        self.__base = read_csv(self.__csv)
        self.__get_predictors_and_classes()
        self.__process()

    def __get_predictors_and_classes(self):
        self.__set_predictors()
        self.__set_classes()

    def __process(self):
        self.__encoder()
        self.__scaler_predictors()

    def __set_predictors(self):
        self.__predictors = self.__base.iloc[:, 0:14].values

    def __set_classes(self):
        self.__classes = self.__base.iloc[:, 14].values

    def __encoder(self):
        self.__encoder_predictors()
        self.__encoder_classes()

    def __encoder_predictors(self):
        self.__predictors = self.__predictors_dummy.fit_transform(self.__predictors).toarray()

    def __encoder_classes(self):
        self.__classes = self.__label_encoder.fit_transform(self.__classes)

    def __scaler_predictors(self):
        self.__predictors = self.__scaler.fit_transform(self.__predictors)

    @property
    def predictors(self):
        return self.__predictors

    @property
    def classes(self):
        return self.__classes


if __name__ == '__main__':
    a = PreProcessCensus()
    a.predictors