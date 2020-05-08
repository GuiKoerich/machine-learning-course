from sklearn.model_selection import train_test_split


class Base:
    __slots__ = ['__training_predictors', '__test_predictors', '__training_classes', '__test_classes', '__test_size',
                 '__process_base']

    def __init__(self, process_base, test_size=25):
        self.__process_base = process_base
        self.__test_size = self.__size(test_size)
        self.__training_predictors, self.__test_predictors, self.__training_classes, self.__test_classes \
            = train_test_split(self.__process_base.predictors, self.__process_base.classes,
                               test_size=self.__test_size, random_state=0)

    @staticmethod
    def __size(value):
        return value / 100

    def training_base(self):
        return self.__training_predictors, self.__training_classes

    def test_base(self):
        return self.__test_predictors, self.__test_classes
