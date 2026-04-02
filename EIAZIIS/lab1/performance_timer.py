import time
from tkinter import messagebox

class PerformanceTimer:
    _start_time = None
    
    @staticmethod
    def start():
        PerformanceTimer._start_time = time.time()
    
    @staticmethod
    def stop(message="Время выполнения"):
        if PerformanceTimer._start_time is None:
            print("Таймер не был запущен!")
            return
        
        elapsed_time = time.time() - PerformanceTimer._start_time
        
        print(f"{message}: {elapsed_time:.4f} секунд")
        messagebox.showinfo("Время обработки текста", f"{message}: {elapsed_time:.4f} секунд")
        PerformanceTimer._start_time = None
