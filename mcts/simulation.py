from abc import ABC, abstractmethod

from .node import Node


class SimulationStrategy(ABC):

    @abstractmethod
    def simulate(self, node: Node) -> float:
        pass