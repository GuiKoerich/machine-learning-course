from naive_bayes import NaiveBayes
from bases import CensusBase


class NaiveBayesCensus(NaiveBayes):
    __slots__ = ['__base']

    def __init__(self, encoder=True, scaler=True, dummy=True):
        self.__base = CensusBase(encoder, scaler, dummy)
        super().__init__(base=self.__base)

    def test_base(self):
        self._test_base()
        print(f'Precisão {self.precision}%')


if __name__ == '__main__':
    NaiveBayesCensus(scaler=False).test_base()
