from .base import Base
from pre_process import PreProcessCensus


class CensusBase(Base):
    __slots__ = []
    __census = PreProcessCensus()

    def __init__(self, test_size=25):
        super().__init__(process_base=self.__census, test_size=test_size)
