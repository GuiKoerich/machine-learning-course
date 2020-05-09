from decision_tree import Tree, TreeCriterion
from bases import CensusBase


class TreeCensus(Tree):
    __slots__ = ['__base']

    def __init__(self, encoder=True, scaler=True, dummy=True, criterion=TreeCriterion.ENTROPY.value):
        self.__base = CensusBase(encoder=encoder, scaler=scaler, dummy=dummy)
        super().__init__(base=self.__base, criterion=criterion)

    def test_base(self):
        self._test_base()
        print(f'Precisão {self.precision}%')


if __name__ == '__main__':
    TreeCensus().test_base()
