from sklearn.model_selection import train_test_split
from pre_process import PreProcessCredit


class CreditBase:
    __slots__ = ['__training_predictors', '__test_predictors', '__training_classes', '__test_classes', '__test_size']
    __credit = PreProcessCredit()

    def __init__(self, test_size=25):
        self.__test_size = self.__size(test_size)
        self.__training_predictors, self.__test_predictors, self.__training_classes, self.__test_classes \
            = train_test_split(self.__credit.predictors, self.__credit.classes,
                               test_size=self.__test_size, random_state=0)

    @staticmethod
    def __size(value):
        return value / 100

    def training_base(self):
        return self.__training_predictors, self.__training_classes

    def test_base(self):
        return self.__test_predictors, self.__test_classes


if __name__ == '__main__':
    c = CreditBase()
    a, b = c.test_base()
    print(a)
    print(b)