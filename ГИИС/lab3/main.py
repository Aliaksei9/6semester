import datetime
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from math import floor, sqrt

from PySide6 import QtCore, QtGui, QtWidgets

from constants import *
@dataclass
class PointStep:
    x: int
    y: int
    alpha: float
    info: str


def row_vec_mat_mul(vec, mat):
    """Умножение вектора-строки длины 4 на матрицу 4x4.
       Возвращает вектор-строку длины 4."""
    res = [0.0] * 4
    for j in range(4):
        s = 0.0
        for i in range(4):
            s += vec[i] * mat[i][j]
        res[j] = s
    return res

def dot_with_geom(coeffs, geom):
    """Вычисляет координаты точки как скалярное произведение вектора коэффициентов
       и столбцов матрицы геометрии (список из 4 кортежей (x, y))."""
    x = sum(coeffs[i] * geom[i][0] for i in range(4))
    y = sum(coeffs[i] * geom[i][1] for i in range(4))
    return x, y


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
        if steps == 0:
            return [PointStep(x0, y0, 1.0, "Point")]
        xi, yi = dx / steps, dy / steps
        x, y = float(x0), float(y0)
        for _ in range(steps + 1):
            pts.append(PointStep(round(int(x + 0.5 * sgn(xi))), round(int(y + 0.5 * sgn(yi))), 1.0, "DDA"))
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
        if steep:
            x0, y0, x1, y1 = y0, x0, y1, x1
        if x0 > x1:
            x0, x1, y0, y1 = x1, x0, y1, y0
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


# ==================== СТРАТЕГИИ ДЛЯ КРИВЫХ ВТОРОГО ПОРЯДКА ====================
class CurveStrategy(ABC):
    @abstractmethod
    def calculate(self, cx: int, cy: int, param1: int, param2: int) -> List[PointStep]:
        pass


class CircleBresenhamStrategy(CurveStrategy):
    def calculate(self, cx: int, cy: int, r: int, param2: int = 0) -> List[PointStep]:
        pts = []
        x, y = 0, r
        delta = 2 - 2 * r
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
        delta = b * b + a * a - 2 * a * b
        pts.extend(self._mirror_points_ellipse(cx, cy, x, y, 1.0, f"delta={delta}"))
        while a * a * (y - 0.5) > b * b * (x + 1):
            if delta < 0:
                delta_h = 2 * (delta + a * a * y) - 1
                if delta_h <= 0:
                    x += 1
                    delta += b * b * (2 * x + 1)
                else:
                    x += 1
                    y -= 1
                    delta += b * b * (2 * x + 1) + a * a * (1 - 2 * y)
            elif delta > 0:
                delta_v = 2 * (delta - b * b * x) - 1
                if delta_v <= 0:
                    x += 1
                    y -= 1
                    delta += b * b * (2 * x + 1) + a * a * (1 - 2 * y)
                else:
                    y -= 1
                    delta += a * a * (1 - 2 * y)
            else:
                x += 1
                y -= 1
                delta += b * b * (2 * x + 1) + a * a * (1 - 2 * y)
            pts.extend(self._mirror_points_ellipse(cx, cy, x, y, 1.0, f"delta={delta}"))
        while y > 0:
            if delta < 0:
                delta_h = 2 * (delta + a * a * y) - 1
                if delta_h <= 0:
                    x += 1
                    delta += b * b * (2 * x + 1)
                else:
                    x += 1
                    y -= 1
                    delta += b * b * (2 * x + 1) + a * a * (1 - 2 * y)
            elif delta > 0:
                delta_v = 2 * (delta - b * b * x) - 1
                if delta_v <= 0:
                    x += 1
                    y -= 1
                    delta += b * b * (2 * x + 1) + a * a * (1 - 2 * y)
                else:
                    y -= 1
                    delta += a * a * (1 - 2 * y)
            else:
                x += 1
                y -= 1
                delta += b * b * (2 * x + 1) + a * a * (1 - 2 * y)
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
        a2, b2 = a * a, b * b
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
        if direction == 0:  # вверх
            return [PointStep(cx + x, cy + y, 1.0, ""), PointStep(cx - x, cy + y, 1.0, "")]
        elif direction == 1:  # вниз
            return [PointStep(cx + x, cy - y, 1.0, ""), PointStep(cx - x, cy - y, 1.0, "")]
        elif direction == 2:  # вправо
            return [PointStep(cx + y, cy + x, 1.0, ""), PointStep(cx + y, cy - x, 1.0, "")]
        else:  # влево
            return [PointStep(cx - y, cy + x, 1.0, ""), PointStep(cx - y, cy - x, 1.0, "")]


# ==================== ПАРАМЕТРИЧЕСКИЕ КРИВЫЕ ====================
class ParamCurveStrategy(ABC):
    @abstractmethod
    def calculate(self, points: List[Tuple[int, int]], **kwargs) -> List[PointStep]:
        pass


class HermiteStrategy(ParamCurveStrategy):
    def calculate(self, points: List[Tuple[int, int]], **kwargs) -> List[PointStep]:
        if len(points) != 2:
            return []
        P1, P4 = points[0], points[1]
        r1 = kwargs.get('r1', (0, 0))
        r2 = kwargs.get('r2', (0, 0))
        print(r1)
        # Матрица геометрии: [P1, P4, r1, r2]
        geom = [P1, P4, r1, r2]
        steps = []
        for i in range(1001):
            t = i / 1000.0
            T = [t**3, t**2, t, 1.0]
            coeffs = row_vec_mat_mul(T, M_HERMITE)
            x, y = dot_with_geom(coeffs, geom)
            steps.append(PointStep(int(round(x)), int(round(y)), 1.0, "Hermite"))
        return steps


class BezierStrategy(ParamCurveStrategy):
    def calculate(self, points: List[Tuple[int, int]], **kwargs) -> List[PointStep]:
        if len(points) != 4:
            return []
        geom = points  # [P1, P2, P3, P4]
        steps = []
        for i in range(1001):
            t = i / 1000.0
            T = [t**3, t**2, t, 1.0]
            coeffs = row_vec_mat_mul(T, M_BEZIER)
            x, y = dot_with_geom(coeffs, geom)
            steps.append(PointStep(int(round(x)), int(round(y)), 1.0, "Bezier"))
        return steps


class BSplineStrategy(ParamCurveStrategy):
    def calculate(self, points: List[Tuple[int, int]], **kwargs) -> List[PointStep]:
        if len(points) < 4:
            return []
        steps = []
        for i in range(len(points) - 3):
            geom = points[i:i+4]  # [P_i-1, P_i, P_i+1, P_i+2]
            for j in range(1000):
                t = j / 1000.0
                T = [t**3, t**2, t, 1.0]
                coeffs = row_vec_mat_mul(T, M_BSPLINE)
                x_raw, y_raw = dot_with_geom(coeffs, geom)
                x = x_raw / 6.0
                y = y_raw / 6.0
                steps.append(PointStep(int(round(x)), int(round(y)), 1.0, "BSpline"))
        return steps

# ==================== ДИАЛОГ ДЛЯ ВЕКТОРОВ ЭРМИТА ====================
class HermiteDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Векторы касательных (Эрмит)")
        layout = QtWidgets.QGridLayout(self)

        layout.addWidget(QtWidgets.QLabel("R1 (dx, dy):"), 0, 0)
        self.r1_dx = QtWidgets.QSpinBox()
        self.r1_dx.setRange(-100, 100)
        self.r1_dx.setValue(1)
        layout.addWidget(self.r1_dx, 0, 1)
        self.r1_dy = QtWidgets.QSpinBox()
        self.r1_dy.setRange(-100, 100)
        self.r1_dy.setValue(1)
        layout.addWidget(self.r1_dy, 0, 2)

        layout.addWidget(QtWidgets.QLabel("R2 (dx, dy):"), 1, 0)
        self.r2_dx = QtWidgets.QSpinBox()
        self.r2_dx.setRange(-100, 100)
        self.r2_dx.setValue(1)
        layout.addWidget(self.r2_dx, 1, 1)
        self.r2_dy = QtWidgets.QSpinBox()
        self.r2_dy.setRange(-100, 100)
        self.r2_dy.setValue(1)
        layout.addWidget(self.r2_dy, 1, 2)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box, 2, 0, 1, 3)

    def get_vectors(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return ((self.r1_dx.value(), self.r1_dy.value()),
                (self.r2_dx.value(), self.r2_dy.value()))


# ==================== ЛОГГЕР ====================
class StepLogger:
    FILENAME = LOG_FILENAME

    @classmethod
    def init_file(cls):
        with open(cls.FILENAME, "w", encoding="utf-8") as f:
            f.write(f"=== ОТЧЕТ {datetime.datetime.now()} ===\n\n")

    @classmethod
    def log_result(cls, name, p1, p2, steps):
        with open(cls.FILENAME, "a", encoding="utf-8") as f:
            f.write(f"Алгоритм: {name} | {p1} -> {p2}\n")
            f.write(
                f"{'№':<{LOG_NUM_WIDTH}} | {'X':<{LOG_X_WIDTH}} | {'Y':<{LOG_Y_WIDTH}} | {'Alpha':<{LOG_ALPHA_WIDTH}} | {'Info'}\n")
            f.write("-" * LOG_SEPARATOR_LENGTH + "\n")
            for i, s in enumerate(steps):
                f.write(
                    f"{i:<{LOG_NUM_WIDTH}} | {s.x:<{LOG_X_WIDTH}} | {s.y:<{LOG_Y_WIDTH}} | {s.alpha:<{LOG_ALPHA_WIDTH}.{LOG_ALPHA_PRECISION}f} | {s.info}\n")
            f.write("=" * LOG_SEPARATOR_LENGTH + "\n\n")


# ==================== МЕНЕДЖЕРЫ ====================
class LineManager:
    def __init__(self, canvas):
        self.canvas = canvas
        self.strategy: LineStrategy = DDAStrategy()

    def set_strategy(self, strategy: LineStrategy):
        self.strategy = strategy

    def calculate_steps(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> List[PointStep]:
        return self.strategy.calculate(p1[0], p1[1], p2[0], p2[1])

    def get_name(self) -> str:
        return self.strategy.__class__.__name__


class CurveManager:
    def __init__(self, canvas):
        self.canvas = canvas
        self.strategy: CurveStrategy = CircleBresenhamStrategy()

    def set_strategy(self, strategy: CurveStrategy):
        self.strategy = strategy

    def calculate_steps(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> List[PointStep]:
        cx, cy = p1
        px, py = p2
        dx, dy = px - cx, py - cy
        name = self.strategy.__class__.__name__
        if name == "CircleBresenhamStrategy":
            r = int(sqrt(dx ** 2 + dy ** 2))
            return self.strategy.calculate(cx, cy, r)
        elif name == "EllipseStrategy":
            return self.strategy.calculate(cx, cy, abs(dx), abs(dy))
        elif name == "HyperbolaStrategy":
            is_vertical = abs(dy) > abs(dx)
            return self.strategy.calculate(cx, cy, max(1, abs(dx)), max(1, abs(dy)), is_vertical)
        elif name == "ParabolaStrategy":
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


# ==================== РЕНДЕРЕР СЕТКИ ====================
class GridRenderer:
    def draw(self, p: QtGui.QPainter, image_size: QtCore.QSize, scale: float):
        h = image_size.height()
        w = image_size.width()

        p.setPen(QtGui.QPen(QtGui.QColor(GRID_COLOR_R, GRID_COLOR_G, GRID_COLOR_B), 0))
        for i in range(0, w + 1, GRID_STEP):
            p.drawLine(i, 0, i, -h)
        for i in range(0, h + 1, GRID_STEP):
            p.drawLine(0, -i, w, -i)

        p.setPen(QtGui.QPen(QtCore.Qt.blue, AXIS_THICKNESS / scale))
        p.drawLine(0, 0, w, 0)
        p.drawLine(0, 0, 0, -h)

        font = p.font()
        font.setPointSizeF(LABEL_FONT_SIZE / scale)
        p.setFont(font)
        p.setPen(QtCore.Qt.white)

        for i in range(LABEL_STEP, w, LABEL_STEP):
            p.drawText(i + X_LABEL_OFFSET_X / scale, X_LABEL_OFFSET_Y / scale, str(i))
        for i in range(LABEL_STEP, h, LABEL_STEP):
            p.drawText(Y_LABEL_OFFSET_X / scale, -i + Y_LABEL_OFFSET_Y / scale, str(i))

        font.setBold(True)
        font.setPointSizeF(AXIS_LABEL_FONT_SIZE / scale)
        p.setFont(font)
        p.setPen(QtCore.Qt.white)
        p.drawText(Y_AXIS_LABEL_OFFSET_X / scale, -h + Y_AXIS_LABEL_OFFSET_Y / scale, "Y")
        p.drawText(w + X_AXIS_LABEL_OFFSET_X / scale, X_AXIS_LABEL_OFFSET_Y / scale, "X")


# ==================== ХОЛСТ ====================
class Canvas(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(CANVAS_MIN_WIDTH, CANVAS_MIN_HEIGHT)
        self.canvas_size = QtCore.QSize(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.image = QtGui.QImage(self.canvas_size, QtGui.QImage.Format_ARGB32)
        self.image.fill(QtCore.Qt.white)

        self.line_manager = LineManager(self)
        self.curve_manager = CurveManager(self)
        self.grid_renderer = GridRenderer()

        self.start_pt: Optional[Tuple[int, int]] = None
        self.current_end: Optional[Tuple[int, int]] = None
        self.mode = "LINE"
        self.is_debug = False
        self.scale = 1.0
        self.offset = QtCore.QPointF(INITIAL_OFFSET_X, INITIAL_OFFSET_Y)
        self.last_pos = QtCore.QPointF()
        self.pan_start = None

        self.anim_steps: List[PointStep] = []
        self.anim_idx = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._on_anim)
        StepLogger.init_file()

        # Параметрические кривые
        self.param_strategy: Optional[ParamCurveStrategy] = None
        self.param_points: List[Tuple[int, int]] = []
        self.param_vectors: Dict[str, Tuple[int, int]] = {}
        self.dragging_index: int = -1

    def set_line_strategy(self, strategy: LineStrategy):
        self.line_manager.set_strategy(strategy)
        self.param_strategy = None

    def set_curve_strategy(self, strategy: CurveStrategy):
        self.curve_manager.set_strategy(strategy)
        self.param_strategy = None

    def set_param_strategy(self, strategy: Optional[ParamCurveStrategy]):
        self.param_strategy = strategy
        self.param_points.clear()
        self.param_vectors.clear()
        self.dragging_index = -1
        self.update()

    def add_param_point(self, x: int, y: int):
        """Добавляет опорную точку. Для Эрмита после 2-х точек сразу запрашиваем векторы."""
        if self.param_strategy is None:
            return

        if isinstance(self.param_strategy, HermiteStrategy) and len(self.param_points) >= 2:
            QtWidgets.QMessageBox.information(self, "Информация", "Для кривой Эрмита достаточно двух точек.")
            return
        if isinstance(self.param_strategy, BezierStrategy) and len(self.param_points) >= 4:
            QtWidgets.QMessageBox.information(self, "Информация", "Для кривой Безье достаточно четырёх точек.")
            return

        self.param_points.append((x, y))

        # Для Эрмита: если добавлена вторая точка, сразу запрашиваем векторы
        if isinstance(self.param_strategy, HermiteStrategy) and len(self.param_points) == 2:
            QtCore.QTimer.singleShot(100, self._ask_hermite_vectors)

        self.update()

    def update_param_point(self, index: int, x: int, y: int):
        """Обновляет координаты опорной точки при перетаскивании."""
        if 0 <= index < len(self.param_points):
            self.param_points[index] = (x, y)
            self.update()

    def _find_closest_param_point(self, pos: Tuple[int, int]) -> Tuple[int, float]:
        min_dist = float('inf')
        idx = -1
        for i, pt in enumerate(self.param_points):
            dx = pt[0] - pos[0]
            dy = pt[1] - pos[1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist:
                min_dist = dist_sq
                idx = i
        return idx, sqrt(min_dist)

    def _ask_hermite_vectors(self):
        """Запрос векторов касательных для кривой Эрмита."""
        if len(self.param_points) < 2:
            return

        dialog = HermiteDialog(self)
        # Устанавливаем значения по умолчанию из существующих векторов (если есть)
        if 'r1' in self.param_vectors:
            dialog.r1_dx.setValue(self.param_vectors['r1'][0])
            dialog.r1_dy.setValue(self.param_vectors['r1'][1])
        if 'r2' in self.param_vectors:
            dialog.r2_dx.setValue(self.param_vectors['r2'][0])
            dialog.r2_dy.setValue(self.param_vectors['r2'][1])

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            r1, r2 = dialog.get_vectors()
            self.param_vectors['r1'] = r1
            self.param_vectors['r2'] = r2
        else:
            # Если пользователь отменил, но точек уже 2, оставляем векторы по умолчанию
            if 'r1' not in self.param_vectors:
                self.param_vectors['r1'] = (1, 1)
            if 'r2' not in self.param_vectors:
                self.param_vectors['r2'] = (1, 1)

        self.update()

    def _build_param_curve(self):
        """Фиксирует текущую параметрическую кривую на холсте и сбрасывает опорные точки."""
        if self.param_strategy is None:
            return

        min_pts = self._min_points_for_strategy()
        if len(self.param_points) < min_pts:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Нужно {min_pts} точек")
            return

        # Для Эрмита проверяем наличие векторов
        if isinstance(self.param_strategy, HermiteStrategy):
            if 'r1' not in self.param_vectors or 'r2' not in self.param_vectors:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала задайте векторы касательных")
                return

        # Вычисляем точки кривой
        steps = self.param_strategy.calculate(self.param_points, **self.param_vectors)

        if steps:
            # Рисуем на постоянное изображение (холст)
            self._draw_to_img(steps)

            # Логируем результат
            name = self.param_strategy.__class__.__name__
            if self.param_points:
                StepLogger.log_result(name, self.param_points[0], self.param_points[-1], steps)

            # СБРАСЫВАЕМ опорные точки и векторы
            self.param_points.clear()
            self.param_vectors.clear()
            self.dragging_index = -1

            # Обновляем отображение
            self.update()

            QtWidgets.QMessageBox.information(self, "Успех",
                                              "Кривая построена и зафиксирована на холсте.\n"
                                              "Можно начинать строить новую кривую.")

    def _min_points_for_strategy(self):
        if self.param_strategy is None:
            return 0
        if isinstance(self.param_strategy, HermiteStrategy):
            return 2
        if isinstance(self.param_strategy, BezierStrategy):
            return 4
        if isinstance(self.param_strategy, BSplineStrategy):
            return 4
        return 2

    def _to_logical(self, pos: QtCore.QPointF) -> Tuple[int, int]:
        lx = (pos.x() - self.offset.x()) / self.scale
        ly = -(pos.y() - self.offset.y()) / self.scale
        return int(round(lx)), int(round(ly))

    def _on_anim(self):
        if self.anim_idx < len(self.anim_steps):
            self.anim_idx += 1
            self.update()
        else:
            self.timer.stop()
            self._draw_to_img(self.anim_steps)

    def _draw_to_img(self, steps: List[PointStep]):
        p = QtGui.QPainter(self.image)
        h = self.image.height()
        for s in steps:
            if 0 <= s.x < self.canvas_size.width() and 0 <= (h - 1 - s.y) < self.canvas_size.height():
                p.setPen(QtGui.QColor(0, 0, 0, int(s.alpha * ALPHA_MAX)))
                p.drawPoint(s.x, (h - 1) - s.y)
        p.end()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            logical_pos = self._to_logical(event.position())
            if self.mode == "PAN":
                self.pan_start = event.pos()
                self.setCursor(QtGui.QCursor(QtCore.Qt.ClosedHandCursor))
                return

            if self.param_strategy is not None and self.mode == "PARAM":
                closest_idx, closest_dist = self._find_closest_param_point(logical_pos)
                if closest_dist < PARAM_POINT_RADIUS:
                    self.dragging_index = closest_idx
                else:
                    self.add_param_point(*logical_pos)
            else:
                if self.start_pt is None:
                    self.start_pt = logical_pos
                else:
                    if self.mode == "LINE":
                        steps = self.line_manager.calculate_steps(self.start_pt, logical_pos)
                    else:  # CURVE
                        steps = self.curve_manager.calculate_steps(self.start_pt, logical_pos)
                    if self.is_debug:
                        self.anim_steps = steps
                        self.anim_idx = 0
                        self.timer.start(ANIM_TIMER_INTERVAL_MS)
                    else:
                        self._draw_to_img(steps)
                    self.start_pt = None
                self.current_end = None
            self.update()

    def mouseMoveEvent(self, event):
        logical_pos = self._to_logical(event.position())
        if self.mode == "PAN" and self.pan_start is not None:
            delta = event.pos() - self.pan_start
            self.offset += delta
            self.pan_start = event.pos()
            self.update()
            return

        if self.param_strategy is not None and self.mode == "PARAM":
            if self.dragging_index != -1:
                self.update_param_point(self.dragging_index, *logical_pos)
        else:
            self.current_end = logical_pos
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.dragging_index != -1:
                self.dragging_index = -1
                self.update()
            elif self.mode == "CURVE" and self.start_pt and self.current_end and self.param_strategy is None:
                steps = self.curve_manager.calculate_steps(self.start_pt, self.current_end)
                if steps:
                    name = self.curve_manager.get_name()
                    StepLogger.log_result(name, self.start_pt, self.current_end, steps)
                    if self.is_debug:
                        self.anim_steps = steps
                        self.anim_idx = 0
                        self.timer.start(ANIM_TIMER_INTERVAL_MS)
                    else:
                        self._draw_to_img(steps)
                self.start_pt = None
                self.current_end = None
                self.update()
            elif self.mode == "PAN":
                self.setCursor(QtCore.Qt.ArrowCursor)
                self.pan_start = None

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.save()
        p.translate(self.offset)
        p.scale(self.scale, self.scale)

        # Белый холст
        h = self.image.height()
        p.drawImage(QtCore.QPoint(0, -h), self.image)

        # Сетка, оси, подписи
        self.grid_renderer.draw(p, self.canvas_size, self.scale)

        # Опорные точки параметрических кривых (синие)
        if self.param_strategy is not None and self.param_points:
            p.setPen(QtGui.QPen(QtCore.Qt.blue, 5 / self.scale))
            for (px, py) in self.param_points:
                p.drawPoint(px, -py)

        # Предпросмотр параметрической кривой (зелёный)
        if self.param_strategy is not None:
            if len(self.param_points) >= self._min_points_for_strategy():
                # Для Эрмита проверяем наличие векторов
                if isinstance(self.param_strategy, HermiteStrategy):
                    if 'r1' in self.param_vectors and 'r2' in self.param_vectors:
                        steps = self.param_strategy.calculate(self.param_points, **self.param_vectors)
                        if steps:
                            p.setPen(QtGui.QPen(QtCore.Qt.green, 2 / self.scale))
                            for s in steps:
                                p.drawPoint(s.x, -s.y)
                else:
                    # Для Безье и B-сплайна
                    steps = self.param_strategy.calculate(self.param_points, **self.param_vectors)
                    if steps:
                        p.setPen(QtGui.QPen(QtCore.Qt.green, 2 / self.scale))
                        for s in steps:
                            p.drawPoint(s.x, -s.y)

        # Анимация (пошаговый режим)
        if self.timer.isActive():
            p.setPen(QtGui.QPen(QtCore.Qt.red, ANIM_PEN_THICKNESS / self.scale))
            for i in range(self.anim_idx):
                s = self.anim_steps[i]
                p.drawPoint(s.x, -s.y)

        # Стартовая точка (для отрезков и обычных кривых)
        if self.start_pt and self.param_strategy is None:
            p.setPen(QtGui.QPen(QtCore.Qt.red, START_PT_PEN_THICKNESS / self.scale))
            p.drawPoint(self.start_pt[0], -self.start_pt[1])

        # Preview для обычных кривых второго порядка
        if self.mode == "CURVE" and self.start_pt and self.current_end and self.param_strategy is None:
            steps = self.curve_manager.calculate_steps(self.start_pt, self.current_end)
            if steps:
                p.setPen(QtGui.QPen(QtCore.Qt.green, ANIM_PEN_THICKNESS / self.scale))
                for s in steps:
                    p.drawPoint(s.x, -s.y)

        p.restore()


# ==================== ГЛАВНОЕ ОКНО ====================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Line Editor: Cartesian System")
        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)

        menubar = self.menuBar()
        line_menu = menubar.addMenu("Отрезки")
        curve_menu = menubar.addMenu("Линии второго порядка")
        param_menu = menubar.addMenu("Параметрические кривые")

        tools_tb = self.addToolBar("Tools")
        tools_tb.setObjectName("Tools")

        # Режимы
        act_line = QtGui.QAction("Отрезки", self, checkable=True)
        act_line.setChecked(True)
        act_line.triggered.connect(lambda: setattr(self.canvas, 'mode', "LINE"))

        act_curve = QtGui.QAction("Кривые", self, checkable=True)
        act_curve.triggered.connect(lambda: setattr(self.canvas, 'mode', "CURVE"))

        act_pan = QtGui.QAction("Рука", self, checkable=True)
        act_pan.triggered.connect(lambda: setattr(self.canvas, 'mode', "PAN"))

        act_param = QtGui.QAction("Параметрические", self, checkable=True)
        act_param.triggered.connect(lambda: setattr(self.canvas, 'mode', "PARAM"))

        mode_group = QtGui.QActionGroup(self)
        mode_group.addAction(act_line)
        mode_group.addAction(act_curve)
        mode_group.addAction(act_pan)
        mode_group.addAction(act_param)

        tools_tb.addActions([act_line, act_curve, act_pan, act_param])
        tools_tb.addSeparator()

        # Панель отрезков
        line_tb = self.addToolBar("Отрезки")
        line_tb.setObjectName("Отрезки")
        self.line_strats = {
            "ЦДА": DDAStrategy(),
            "Брезенхем": BresenhamStrategy(),
            "Ву": WuStrategy()
        }
        line_strat_group = QtGui.QActionGroup(self)
        line_strat_group.setExclusive(True)
        for name, strategy in self.line_strats.items():
            act = QtGui.QAction(name, self, checkable=True)
            act.triggered.connect(lambda checked, s=strategy: self._set_line_strategy(s) if checked else None)
            line_strat_group.addAction(act)
            line_menu.addAction(act)
            line_tb.addAction(act)
            if name == "ЦДА":
                act.setChecked(True)
        line_tb.addSeparator()

        # Панель кривых второго порядка
        curve_tb = self.addToolBar("Линии второго порядка")
        curve_tb.setObjectName("Линии второго порядка")
        self.curve_strats = {
            "Окружность": CircleBresenhamStrategy(),
            "Эллипс": EllipseStrategy(),
            "Гипербола": HyperbolaStrategy(),
            "Парабола": ParabolaStrategy()
        }
        curve_strat_group = QtGui.QActionGroup(self)
        curve_strat_group.setExclusive(True)
        for name, strategy in self.curve_strats.items():
            act = QtGui.QAction(name, self, checkable=True)
            act.triggered.connect(lambda checked, s=strategy: self._set_curve_strategy(s) if checked else None)
            curve_strat_group.addAction(act)
            curve_menu.addAction(act)
            curve_tb.addAction(act)
            if name == "Окружность":
                act.setChecked(True)
        curve_tb.addSeparator()

        # Панель параметрических кривых
        param_tb = self.addToolBar("Параметрические кривые")
        param_tb.setObjectName("Параметрические кривые")
        self.param_strats = {
            "Эрмит": HermiteStrategy(),
            "Безье": BezierStrategy(),
            "B-сплайн": BSplineStrategy()
        }
        param_group = QtGui.QActionGroup(self)
        param_group.setExclusive(True)
        for name, strategy in self.param_strats.items():
            act = QtGui.QAction(name, self, checkable=True)
            act.triggered.connect(lambda checked, s=strategy: self._set_param_strategy(s) if checked else None)
            param_group.addAction(act)
            param_menu.addAction(act)
            param_tb.addAction(act)

        self.build_param_btn = QtGui.QAction("Построить параметрическую кривую", self)
        self.build_param_btn.triggered.connect(self._build_param)
        param_tb.addAction(self.build_param_btn)
        self.build_param_btn.setEnabled(False)
        param_tb.addSeparator()

        # Общие инструменты
        chk = QtWidgets.QCheckBox("Пошагово")
        chk.toggled.connect(lambda v: setattr(self.canvas, 'is_debug', v))
        tools_tb.addWidget(chk)

        tools_tb.addAction("Zoom +", lambda: self._zoom(ZOOM_IN_FACTOR))
        tools_tb.addAction("Zoom -", lambda: self._zoom(ZOOM_OUT_FACTOR))
        tools_tb.addAction("Очистить", self._reset)

        self._set_line_strategy(self.line_strats["ЦДА"])
        self._set_curve_strategy(self.curve_strats["Окружность"])
        self.canvas.set_param_strategy(None)

    def _set_line_strategy(self, strategy: LineStrategy):
        self.canvas.set_line_strategy(strategy)

    def _set_curve_strategy(self, strategy: CurveStrategy):
        self.canvas.set_curve_strategy(strategy)

    def _set_param_strategy(self, strategy: ParamCurveStrategy):
        self.canvas.set_param_strategy(strategy)
        self.build_param_btn.setEnabled(strategy is not None)

    def _build_param(self):
        """Построение параметрической кривой по кнопке (фиксация на холсте)."""
        if self.canvas.param_strategy is None:
            return

        # Для Эрмита проверяем наличие векторов
        if isinstance(self.canvas.param_strategy, HermiteStrategy):
            if len(self.canvas.param_points) >= 2:
                if 'r1' not in self.canvas.param_vectors or 'r2' not in self.canvas.param_vectors:
                    # Если векторов нет, но есть 2 точки - запрашиваем
                    self.canvas._ask_hermite_vectors()
                    # После закрытия диалога, если векторы заданы, можно строить
                    if 'r1' in self.canvas.param_vectors and 'r2' in self.canvas.param_vectors:
                        self.canvas._build_param_curve()
                    return

        # Для всех остальных случаев просто фиксируем на холсте
        self.canvas._build_param_curve()

    def _zoom(self, f: float):
        self.canvas.scale *= f
        self.canvas.update()

    def _reset(self):
        self.canvas.image.fill(QtCore.Qt.white)
        self.canvas.update()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
    win.show()
    sys.exit(app.exec())