from abc import ABC, abstractmethod
from typing import List
from math import sqrt
from constants import *
from line_strategies import PointStep

class CurveStrategy(ABC):
    @abstractmethod
    def calculate(self, cx: int, cy: int, param1: int, param2: int) -> List[PointStep]:
        pass

class CircleBresenhamStrategy(CurveStrategy):
    def calculate(self, cx: int, cy: int, r: int, param2: int = 0) -> List[PointStep]:
        pts = []
        x, y = 0, r
        delta = 2 - 2 * r
        error = 0
        pts.extend(self._mirror_points(cx, cy, x, y, 1.0, f"delta={delta}"))
        while y >= x:
            if delta < 0:
                delta_h = 2 * (delta + y) - 1
                if delta_h <= 0:
                    x += 1
                    delta += 2 * x + 1
                else:
                    x += 1
                    y -= 1
                    delta += 2 * (x - y + 1)
            elif delta > 0:
                delta_v = 2 * (delta - x) - 1
                if delta_v <= 0:
                    x += 1
                    y -= 1
                    delta += 2 * (x - y + 1)
                else:
                    y -= 1
                    delta += -2 * y + 1
            else:
                x += 1
                y -= 1
                delta += 2 * (x - y + 1)
            pts.extend(self._mirror_points(cx, cy, x, y, 1.0, f"delta={delta}"))
        return pts

    def _mirror_points(self, cx, cy, x, y, alpha, info):
        return [
            PointStep(cx + x, cy + y, alpha, info),
            PointStep(cx - x, cy + y, alpha, info),
            PointStep(cx + x, cy - y, alpha, info),
            PointStep(cx - x, cy - y, alpha, info),
            PointStep(cx + y, cy + x, alpha, info),
            PointStep(cx - y, cy + x, alpha, info),
            PointStep(cx + y, cy - x, alpha, info),
            PointStep(cx - y, cy - x, alpha, info)
        ]

class EllipseStrategy(CurveStrategy):
    def calculate(self, cx: int, cy: int, a: int, b: int) -> List[PointStep]:
        pts = []
        x, y = 0, b
        delta = b**2 + a**2 - 2*a*b
        pts.extend(self._mirror_points_ellipse(cx, cy, x, y, 1.0, f"delta={delta}"))
        while a**2 * (y - 0.5) > b**2 * (x + 1):
            if delta < 0:
                delta_h = 2 * (delta + a**2 * y) - 1
                if delta_h <= 0:
                    x += 1
                    delta += b**2 * (2 * x + 1)
                else:
                    x += 1
                    y -= 1
                    delta += b**2 * (2 * x + 1) + a**2 * (1 - 2 * y)
            elif delta > 0:
                delta_v = 2 * (delta - b**2 * x) - 1
                if delta_v <= 0:
                    x += 1
                    y -= 1
                    delta += b**2 * (2 * x + 1) + a**2 * (1 - 2 * y)
                else:
                    y -= 1
                    delta += a**2 * (1 - 2 * y)
            else:
                x += 1
                y -= 1
                delta += b**2 * (2 * x + 1) + a**2 * (1 - 2 * y)
            pts.extend(self._mirror_points_ellipse(cx, cy, x, y, 1.0, f"delta={delta}"))
        while y > 0:
            if delta < 0:
                delta_h = 2 * (delta + a**2 * y) - 1
                if delta_h <= 0:
                    x += 1
                    delta += b**2 * (2 * x + 1)
                else:
                    x += 1
                    y -= 1
                    delta += b**2 * (2 * x + 1) + a**2 * (1 - 2 * y)
            elif delta > 0:
                delta_v = 2 * (delta - b**2 * x) - 1
                if delta_v <= 0:
                    x += 1
                    y -= 1
                    delta += b**2 * (2 * x + 1) + a**2 * (1 - 2 * y)
                else:
                    y -= 1
                    delta += a**2 * (1 - 2 * y)
            else:
                x += 1
                y -= 1
                delta += b**2 * (2 * x + 1) + a**2 * (1 - 2 * y)
            pts.extend(self._mirror_points_ellipse(cx, cy, x, y, 1.0, f"delta={delta}"))
        return pts

    def _mirror_points_ellipse(self, cx, cy, x, y, alpha, info):
        return [
            PointStep(cx + x, cy + y, alpha, info),
            PointStep(cx - x, cy + y, alpha, info),
            PointStep(cx + x, cy - y, alpha, info),
            PointStep(cx - x, cy - y, alpha, info)
        ]

class HyperbolaStrategy(CurveStrategy):
    def calculate(self, cx: int, cy: int, a: int, b: int, is_vertical: bool = False) -> List[PointStep]:
        pts = []
        if a == 0 or b == 0:
            return pts
        swap = is_vertical
        if swap:
            a, b = b, a
        x = a
        y = 0
        a2 = a * a
        b2 = b * b
        pts.extend(self._mirror(cx, cy, x, y, is_vertical))
        p = b2 * (x + 0.5) ** 2 - a2 * (y + 1) ** 2 - a2 * b2
        while b2 * x >= a2 * y and x <= CANVAS_WIDTH and y <= CANVAS_HEIGHT:
            if p >= 0:
                p += -a2 * (2 * y + 3)
            else:
                p += -a2 * (2 * y + 3) + b2 * (2 * x + 2)
                x += 1
            y += 1
            pts.extend(self._mirror(cx, cy, x, y, is_vertical))
        p = -b2 * (x + 1) ** 2 + a2 * (y + 0.5) ** 2 + a2 * b2
        while x <= CANVAS_WIDTH and y <= CANVAS_HEIGHT:
            if p < 0:
                p += -b2 * (2 * x + 3) + a2 * (2 * y + 2)
                y += 1
            else:
                p += -b2 * (2 * x + 3)
            x += 1
            pts.extend(self._mirror(cx, cy, x, y, is_vertical))
        return pts

    def _mirror(self, cx, cy, x, y, is_vertical):
        if is_vertical:
            x, y = y, x
        return [
            PointStep(cx + x, cy + y, 1.0, ""),
            PointStep(cx + x, cy - y, 1.0, ""),
            PointStep(cx - x, cy + y, 1.0, ""),
            PointStep(cx - x, cy - y, 1.0, "")
        ]

class ParabolaStrategy(CurveStrategy):
    def calculate(self, cx: int, cy: int, p: int, direction: int = 0) -> List[PointStep]:
        pts = []
        if p <= 0:
            return pts
        x, y = 0, 0
        d = 1 - 2 * p
        while x <= 2 * p and x <= CANVAS_WIDTH:
            pts.extend(self._add_points(cx, cy, x, y, direction))
            if d > 0:
                y += 1
                d -= 4 * p
            x += 1
            d += 2 * x + 1
            if y > CANVAS_HEIGHT:
                break
        d = (x + 0.5) ** 2 - 4 * p * (y + 1)
        while y <= CANVAS_HEIGHT and x <= CANVAS_WIDTH:
            pts.extend(self._add_points(cx, cy, x, y, direction))
            if d < 0:
                x += 1
                d += 2 * x
            y += 1
            d -= 4 * p
        return pts

    def _add_points(self, cx, cy, x, y, direction):
        if direction == 0:
            return [PointStep(cx + x, cy + y, 1.0, ""), PointStep(cx - x, cy + y, 1.0, "")]
        elif direction == 1:
            return [PointStep(cx + x, cy - y, 1.0, ""), PointStep(cx - x, cy - y, 1.0, "")]
        elif direction == 2:
            return [PointStep(cx + y, cy + x, 1.0, ""), PointStep(cx + y, cy - x, 1.0, "")]
        else:
            return [PointStep(cx - y, cy + x, 1.0, ""), PointStep(cx - y, cy - x, 1.0, "")]