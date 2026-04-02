from typing import List, Tuple
from math import sqrt
from constants import *
from line_strategies import LineStrategy, PointStep, DDAStrategy
from curve_strategies import CurveStrategy, CircleBresenhamStrategy

class LineManager:
    def __init__(self, canvas):
        self.canvas = canvas
        self.strategy = DDAStrategy()

    def set_strategy(self, strategy: LineStrategy):
        self.strategy = strategy

    def calculate_steps(self, p1: tuple[int, int], p2: tuple[int, int]) -> List[PointStep]:
        x0, y0 = p1
        x1, y1 = p2
        return self.strategy.calculate(x0, y0, x1, y1)

    def get_name(self) -> str:
        return self.strategy.__class__.__name__

class CurveManager:
    def __init__(self, canvas):
        self.canvas = canvas
        self.strategy = CircleBresenhamStrategy()

    def set_strategy(self, strategy: CurveStrategy):
        self.strategy = strategy

    def calculate_steps(self, p1: tuple[int, int], p2: tuple[int, int]) -> List[PointStep]:
        cx, cy = p1
        px, py = p2
        dx, dy = px - cx, py - cy
        strategy_name = self.strategy.__class__.__name__
        if strategy_name == "CircleBresenhamStrategy":
            r = int(sqrt(dx ** 2 + dy ** 2))
            return self.strategy.calculate(cx, cy, r)
        elif strategy_name == "EllipseStrategy":
            return self.strategy.calculate(cx, cy, abs(dx), abs(dy))
        elif strategy_name == "HyperbolaStrategy":
            is_vertical = abs(dy) > abs(dx)
            return self.strategy.calculate(cx, cy, max(1, abs(dx)), max(1, abs(dy)), is_vertical)
        elif strategy_name == "ParabolaStrategy":
            if abs(dx) > abs(dy):
                direction = 2 if dx > 0 else 3
                denom = 2 * abs(dx)
                param = max(1, int(dy ** 2 / denom)) if denom != 0 else 1
            else:
                direction = 0 if dy > 0 else 1
                denom = 2 * abs(dy)
                param = max(1, int(dx ** 2 / denom)) if denom != 0 else 1
            return self.strategy.calculate(cx, cy, param, direction)
        return []

    def get_name(self) -> str:
        return self.strategy.__class__.__name__