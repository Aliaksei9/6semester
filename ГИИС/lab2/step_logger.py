import datetime
from constants import *
from line_strategies import PointStep

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
            f.write(f"{'№':<{LOG_NUM_WIDTH}} | {'X':<{LOG_X_WIDTH}} | {'Y':<{LOG_Y_WIDTH}} | {'Alpha':<{LOG_ALPHA_WIDTH}} | {'Info'}\n")
            f.write("-" * LOG_SEPARATOR_LENGTH + "\n")
            for i, s in enumerate(steps):
                f.write(f"{i:<{LOG_NUM_WIDTH}} | {s.x:<{LOG_X_WIDTH}} | {s.y:<{LOG_Y_WIDTH}} | {s.alpha:<{LOG_ALPHA_WIDTH}.{LOG_ALPHA_PRECISION}f} | {s.info}\n")
            f.write("=" * LOG_SEPARATOR_LENGTH + "\n\n")