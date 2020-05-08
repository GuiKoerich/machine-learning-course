from naive_bayes import NaiveBayes
from bases import CensusBase


class NaiveBayesCensus(NaiveBayes):
    __slots__ = []

    __base = CensusBase()

    def __init__(self):
        super().__init__(base=self.__base)

    def test_base(self):
        self._test_base()
        print(f'Precisão {self.precision}%')


if __name__ == '__main__':
    NaiveBayesCensus().test_base()
