from decision_tree import Tree, TreeCriterion
from bases import CreditBase


class TreeCredit(Tree):
    __slots__ = []

    __base = CreditBase()

    def __init__(self, criterion=TreeCriterion.ENTROPY.value):
        super().__init__(base=self.__base, criterion=criterion)

    def test_base(self):
        self._test_base()
        print(f'Precisão {self.precision}%')


if __name__ == '__main__':
    TreeCredit().test_base()
