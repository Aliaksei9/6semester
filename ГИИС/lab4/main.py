import pygame
import numpy as np
import sys
from abc import ABC, abstractmethod

class TransformationStrategy(ABC):
    @abstractmethod
    def get_matrix(self) -> np.ndarray:
        pass

class TranslateStrategy(TransformationStrategy):
    def __init__(self, dx=0.0, dy=0.0, dz=0.0):
        self.dx, self.dy, self.dz = dx, dy, dz
    def get_matrix(self):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [self.dx, self.dy, self.dz, 1]
        ], dtype=float)

class RotateXStrategy(TransformationStrategy):
    def __init__(self, angle_deg=3.0):
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, c, s, 0],
            [0, -s, c, 0],
            [0, 0, 0, 1]
        ], dtype=float)
    def get_matrix(self): return self.matrix

class RotateYStrategy(TransformationStrategy):
    def __init__(self, angle_deg=3.0):
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        self.matrix = np.array([
            [c, 0, -s, 0],
            [0, 1, 0, 0],
            [s, 0, c, 0],
            [0, 0, 0, 1]
        ], dtype=float)
    def get_matrix(self): return self.matrix

class RotateZStrategy(TransformationStrategy):
    def __init__(self, angle_deg=3.0):
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        self.matrix = np.array([
            [c, s, 0, 0],
            [-s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
    def get_matrix(self): return self.matrix

class ScaleStrategy(TransformationStrategy):
    def __init__(self, sx=1.0, sy=1.0, sz=1.0):
        self.sx, self.sy, self.sz = sx, sy, sz
    def get_matrix(self):
        return np.array([
            [self.sx, 0, 0, 0],
            [0, self.sy, 0, 0],
            [0, 0, self.sz, 0],
            [0, 0, 0, 1]
        ], dtype=float)

class ReflectStrategy(TransformationStrategy):
    def get_matrix(self):
        return np.array([
            [-1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1]
        ], dtype=float)

class Model:
    def __init__(self, vertices, edges):
        self.original = np.array([v + [1.0] for v in vertices], dtype=float)
        self.edges = edges
        self.model_matrix = np.eye(4, dtype=float)

    def apply_transformation(self, strategy: TransformationStrategy):
        M = strategy.get_matrix()
        self.model_matrix = self.model_matrix @ M

    def get_transformed_vertices(self):
        return self.original @ self.model_matrix

    def reset(self):
        self.model_matrix = np.eye(4, dtype=float)

class View:
    ORTHO_XY = 0
    ORTHO_XZ = 1
    ORTHO_YZ = 2

    def __init__(self, width=900, height=700):
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Лабораторная 4")
        self.width = width
        self.height = height
        self.cx, self.cy = width//2, height//2
        self.ortho_scale = 150.0
        self.focal = 400

        self.projection_mode = None
        self.perspective_enabled = False

    def set_ortho_projection(self, mode):
        self.projection_mode = mode
        self.perspective_enabled = False

    def toggle_perspective(self):
        self.perspective_enabled = not self.perspective_enabled
        if self.perspective_enabled:
            self.projection_mode = None

    def reset_projection(self):
        self.projection_mode = None
        self.perspective_enabled = False

    def project(self, vertices):
        points = []
        for v in vertices:
            x_h, y_h, z_h, w = v

            divisor = w if abs(w) > 1e-6 else 1e-6
            xc, yc, zc = x_h / divisor, y_h / divisor, z_h / divisor

            if self.projection_mode == View.ORTHO_XY:
                sx = self.cx + xc * self.ortho_scale
                sy = self.cy - yc * self.ortho_scale
            elif self.projection_mode == View.ORTHO_XZ:
                sx = self.cx + xc * self.ortho_scale
                sy = self.cy - zc * self.ortho_scale
            elif self.projection_mode == View.ORTHO_YZ:
                sx = self.cx + yc * self.ortho_scale
                sy = self.cy - zc * self.ortho_scale
            else:
                if self.perspective_enabled:
                    z_view = zc + 5.0
                    if abs(z_view) < 0.1: z_view = 0.1
                    sx = self.cx + (self.focal * xc) / z_view
                    sy = self.cy - (self.focal * yc) / z_view
                else:
                    sx = self.cx + xc * (self.focal / 4.0)
                    sy = self.cy - yc * (self.focal / 4.0)

            points.append((sx, sy))
        return points

    def render(self, model: Model):
        self.screen.fill((10, 10, 20))
        verts = model.get_transformed_vertices()
        proj = self.project(verts)

        for i, j in model.edges:
            if i < len(proj) and j < len(proj):
                pygame.draw.line(self.screen, (0, 255, 255), proj[i], proj[j], 3)

        for p in proj:
            pygame.draw.circle(self.screen, (255, 255, 100), (int(p[0]), int(p[1])), 5)

        # Справка
        font = pygame.font.SysFont("consolas", 16)
        help_lines = [
            "← → ↑ ↓   /   z x  – перемещение",
            "R / T / Y         – поворот X / Y / Z",
            "S / Shift+S       – равномерное масштабирование",
            "U/J / I/K / O/L   – масштаб по осям X/Y/Z",
            "F                 – отражение (Z -> -Z)",
            "P                 – включить/выключить перспективу",
            "1 / 2 / 3         – ортографическая проекция: XY / XZ / YZ",
            "0 / Backspace     – сброс всех преобразований",
            "Q                 – выход"
        ]
        for i, txt in enumerate(help_lines):
            surf = font.render(txt, True, (200, 200, 200))
            self.screen.blit(surf, (10, 10 + i*20))

        info = []
        if self.projection_mode == View.ORTHO_XY: info.append("Ortho: XY")
        elif self.projection_mode == View.ORTHO_XZ: info.append("Ortho: XZ")
        elif self.projection_mode == View.ORTHO_YZ: info.append("Ortho: YZ")
        else: info.append("Projection: standard")

        if self.perspective_enabled:
            info.append("Perspective: ON")
        else:
            info.append("Perspective: OFF")

        for i, txt in enumerate(info):
            surf = font.render(txt, True, (255, 200, 100))
            self.screen.blit(surf, (self.width - 250, 10 + i*20))

        pygame.display.flip()

class Controller:
    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view
        self.running = True
        self.step = 0.3
        self.angle = 3.0
        self.scale_uniform = 1.08
        self.scale_axis = 1.08

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    self.running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.view.set_ortho_projection(View.ORTHO_XY)
                    elif event.key == pygame.K_2:
                        self.view.set_ortho_projection(View.ORTHO_XZ)
                    elif event.key == pygame.K_3:
                        self.view.set_ortho_projection(View.ORTHO_YZ)
                    elif event.key == pygame.K_p:
                        self.view.toggle_perspective()
                    elif event.key in (pygame.K_0, pygame.K_BACKSPACE):
                        self.model.reset()
                        self.view.reset_projection()
                    elif event.key == pygame.K_f:
                        self.model.apply_transformation(ReflectStrategy())
                    else:
                        mods = pygame.key.get_mods()
                        shift = bool(mods & pygame.KMOD_SHIFT)
                        if event.key == pygame.K_s:
                            factor = self.scale_uniform if not shift else 1.0/self.scale_uniform
                            self.model.apply_transformation(ScaleStrategy(factor, factor, factor))
                        elif event.key == pygame.K_u:
                            factor = self.scale_axis if not shift else 1.0/self.scale_axis
                            self.model.apply_transformation(ScaleStrategy(factor, 1.0, 1.0))
                        elif event.key == pygame.K_j:
                            factor = self.scale_axis if not shift else 1.0/self.scale_axis
                            self.model.apply_transformation(ScaleStrategy(1.0/factor, 1.0, 1.0))
                        elif event.key == pygame.K_i:
                            factor = self.scale_axis if not shift else 1.0/self.scale_axis
                            self.model.apply_transformation(ScaleStrategy(1.0, factor, 1.0))
                        elif event.key == pygame.K_k:
                            factor = self.scale_axis if not shift else 1.0/self.scale_axis
                            self.model.apply_transformation(ScaleStrategy(1.0, 1.0/factor, 1.0))
                        elif event.key == pygame.K_o:
                            factor = self.scale_axis if not shift else 1.0/self.scale_axis
                            self.model.apply_transformation(ScaleStrategy(1.0, 1.0, factor))
                        elif event.key == pygame.K_l:
                            factor = self.scale_axis if not shift else 1.0/self.scale_axis
                            self.model.apply_transformation(ScaleStrategy(1.0, 1.0, 1.0/factor))

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.model.apply_transformation(TranslateStrategy(dx=-self.step))
            if keys[pygame.K_RIGHT]:
                self.model.apply_transformation(TranslateStrategy(dx=self.step))
            if keys[pygame.K_UP]:
                self.model.apply_transformation(TranslateStrategy(dy=self.step))
            if keys[pygame.K_DOWN]:
                self.model.apply_transformation(TranslateStrategy(dy=-self.step))
            if keys[pygame.K_z]:
                self.model.apply_transformation(TranslateStrategy(dz=self.step))
            if keys[pygame.K_x]:
                self.model.apply_transformation(TranslateStrategy(dz=-self.step))

            if keys[pygame.K_r]:
                self.model.apply_transformation(RotateXStrategy(self.angle))
            if keys[pygame.K_t]:
                self.model.apply_transformation(RotateYStrategy(self.angle))
            if keys[pygame.K_y]:
                self.model.apply_transformation(RotateZStrategy(self.angle))

            self.view.render(self.model)
            clock.tick(60)

        pygame.quit()
        sys.exit()

def load_object(filename="object.txt"):
    vertices = []
    edges = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [float(x) for x in line.split()]
                if len(parts) == 3:
                    vertices.append(parts)
                elif len(parts) == 2:
                    edges.append((int(parts[0]), int(parts[1])))
    except FileNotFoundError:
        print(f"Файл {filename} не найден. Создаётся асимметричная пирамида.")
        vertices = [
            [-1.0, -1.0, 0.0],
            [ 1.0, -1.0, 0.0],
            [ 1.0,  1.0, 0.0],
            [-1.0,  1.0, 0.0],
            [ 0.5,  0.5, 2.0]
        ]
        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (0,4), (1,4), (2,4), (3,4)
        ]
    return vertices, edges

if __name__ == "__main__":
    pygame.init()
    vertices, edges = load_object("object.txt")
    model = Model(vertices, edges)
    view = View()
    controller = Controller(model, view)
    controller.run()