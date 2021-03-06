from .base import Base
from pre_process import PreProcessCredit


class CreditBase(Base):
    __slots__ = ['__credit']

    def __init__(self, test_size=25):
        self.__credit = PreProcessCredit()
        super().__init__(process_base=self.__credit, test_size=test_size)
