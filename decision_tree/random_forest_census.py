from decision_tree import RandomForest, TreeCriterion
from bases import CensusBase


class RandomForestCensus(RandomForest):
    __slots__ = ['__base']

    def __init__(self, encoder=True, scaler=True, dummy=True, criterion=TreeCriterion.ENTROPY.value, trees=10):
        self.__base = CensusBase(encoder, scaler, dummy)
        super().__init__(self.__base, criterion, trees=trees)

    def test_base(self):
        self._test_base()
        print(f'Precisão {self.precision}%')


if __name__ == '__main__':
    RandomForestCensus(trees=15).test_base()
