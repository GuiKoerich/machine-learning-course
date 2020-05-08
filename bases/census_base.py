from .base import Base
from pre_process import PreProcessCensus


class CensusBase(Base):
    __slots__ = ['__census']

    def __init__(self, encoder=True, scaler=True, dummy=True, test_size=25):
        self.__census = PreProcessCensus(encoder, scaler, dummy)
        super().__init__(process_base=self.__census, test_size=test_size)
