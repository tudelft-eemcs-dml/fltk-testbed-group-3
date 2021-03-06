from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EpochData:
    epoch_id: int
    duration_train: int
    duration_test: int
    loss_train: float
    accuracy: float
    loss: float
    class_precision: Any
    class_recall: Any
    client_id: str = None

    def to_csv_line(self):
        delimeter = ','
        values = self.__dict__.values()
        values = [str(x) for x in values]
        return delimeter.join(values)

@dataclass
class GANEpochData:
    epoch_id: int
    duration_train: int
    duration_test: int
    disc: Any
    client_id: str = None

    def to_csv_line(self):
        delimeter = ','
        values = self.__dict__.values()
        values = [str(x) for x in values if str(x) != 'disc']
        return delimeter.join(values)


@dataclass
class FeGANEpochData:
    epoch_id: int
    duration_train: int
    duration_test: int
    net: Any
    client_id: str = None

    def to_csv_line(self):
        delimeter = ','
        values = self.__dict__.values()
        values = [str(x) for x in values if str(x) != 'net']
        return delimeter.join(values)

