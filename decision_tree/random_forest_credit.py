from decision_tree import RandomForest, TreeCriterion
from bases import CreditBase


class RandomForestCredit(RandomForest):
    __slots__ = []

    __base = CreditBase()

    def __init__(self, criterion=TreeCriterion.ENTROPY.value, trees=10):
        super().__init__(self.__base, criterion, trees=trees)

    def test_base(self):
        self._test_base()
        print(f'Precisão {self.precision}%')


if __name__ == '__main__':
    RandomForestCredit(trees=15).test_base()
