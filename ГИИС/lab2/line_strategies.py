from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from math import floor

@dataclass
class PointStep:
    x: int
    y: int
    alpha: float
    info: str

class LineStrategy(ABC):
    @abstractmethod
    def calculate(self, x0: int, y0: int, x1: int, y1: int) -> List[PointStep]:
        pass

class DDAStrategy(LineStrategy):
    def calculate(self, x0, y0, x1, y1):
        sgn = lambda s: (s > 0) - (s < 0)
        pts = []
        dx, dy = x1 - x0, y1 - y0
        steps = int(max(abs(dx), abs(dy)))
        if steps == 0: return [PointStep(x0, y0, 1.0, "Point")]
        xi, yi = dx / steps, dy / steps
        x, y = float(x0), float(y0)
        for _ in range(steps + 1):
            pts.append(PointStep(round(int(x+0.5*sgn(xi))), round(int(y+0.5*sgn(yi))), 1.0, "DDA"))
            x += xi
            y += yi
        return pts

class BresenhamStrategy(LineStrategy):
    def calculate(self, x0: int, y0: int, x1: int, y1: int) -> List[PointStep]:
        pts = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        if dx >= dy:
            x, y = x0, y0
            e = 2 * dy - dx
            pts.append(PointStep(x, y, 1.0, f"e={e}"))
            for _ in range(dx):
                if e >= 0:
                    y += sy
                    e -= 2 * dx
                x += sx
                e += 2 * dy
                pts.append(PointStep(x, y, 1.0, f"e={e}"))
        else:
            x, y = x0, y0
            e = 2 * dx - dy
            pts.append(PointStep(x, y, 1.0, f"e={e}"))
            for _ in range(dy):
                if e >= 0:
                    x += sx
                    e -= 2 * dy
                y += sy
                e += 2 * dx
                pts.append(PointStep(x, y, 1.0, f"e={e}"))
        return pts

class WuStrategy(LineStrategy):
    def calculate(self, x0, y0, x1, y1):
        def ipart(x):
            return int(floor(x))
        def roundi(x):
            return ipart(x + 0.5)
        def fpart(x):
            return x - floor(x)
        def rfpart(x):
            return 1 - fpart(x)
        pts = []
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep: x0, y0, x1, y1 = y0, x0, y1, x1
        if x0 > x1: x0, x1, y0, y1 = x1, x0, y1, y0
        dx, dy = x1 - x0, y1 - y0
        grad = dy / dx if dx != 0 else 1.0
        xend = roundi(x0)
        yend = y0 + grad * (xend - x0)
        xgap = rfpart(x0 + 0.5)
        px1, py1 = xend, ipart(yend)
        if steep:
            pts.append(PointStep(py1, px1, rfpart(yend) * xgap, "start"))
            pts.append(PointStep(py1 + 1, px1, fpart(yend) * xgap, "start"))
        else:
            pts.append(PointStep(px1, py1, rfpart(yend) * xgap, "start"))
            pts.append(PointStep(px1, py1 + 1, fpart(yend) * xgap, "start"))
        intery = yend + grad
        xend = roundi(x1)
        yend = y1 + grad * (xend - x1)
        xgap = fpart(x1 + 0.5)
        px2, py2 = xend, ipart(yend)
        if steep:
            pts.append(PointStep(py2, px2, rfpart(yend) * xgap, "end"))
            pts.append(PointStep(py2 + 1, px2, fpart(yend) * xgap, "end"))
        else:
            pts.append(PointStep(px2, py2, rfpart(yend) * xgap, "end"))
            pts.append(PointStep(px2, py2 + 1, fpart(yend) * xgap, "end"))
        for x in range(px1 + 1, px2):
            if steep:
                pts.append(PointStep(ipart(intery), x, rfpart(intery), "aa"))
                pts.append(PointStep(ipart(intery) + 1, x, fpart(intery), "aa"))
            else:
                pts.append(PointStep(x, ipart(intery), rfpart(intery), "aa"))
                pts.append(PointStep(x, ipart(intery) + 1, fpart(intery), "aa"))
            intery += grad
        return pts