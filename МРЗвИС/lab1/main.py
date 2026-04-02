import random
import os
import math
import matplotlib.pyplot as plt
import sys

"""
Лабораторная работа 1 по дисциплине 

Выполнил студент группы 321701:
- Хмара Алексей Вячеславович

Вариант 1
- Алгоритм вычисления произведения пары 4-х разрядных чисел умножением с младших разрядов со сдвигом множимого влево

Программирование операции обработки данных и знаний с конвейерной обработкой.
09.03.2026

"""


def clear_console():
    os.system('cls')


def format_bin(value, bits_needed):
    total_len = math.ceil(bits_needed / 4) * 4
    if total_len == 0: total_len = 4
    s = format(value & ((1 << total_len) - 1), f'0{total_len}b')
    return " ".join(s[i:i + 4] for i in range(0, len(s), 4))


class Stage:
    def __init__(self, stage_idx, cycles_required):
        self.idx = stage_idx
        self.cycles_required = cycles_required
        self.current_pair = None
        self.cycles_left = 0
        self.p_sum = 0
        self.p_prod = 0


class Simulation:
    def __init__(self, p, queue, stage_times):
        self.p = p
        self.queue = queue
        self.m = len(queue)
        self.stage_times = stage_times
        self.tact = 0
        self.stages = [Stage(i, t) for i, t in enumerate(stage_times)]
        self.results = []
        self.history = []

        self.ky = 0.0  #
        self.eff = 0.0

    def save_state(self):
        state = {
            'tact': self.tact,
            'queue': [dict(p) for p in self.queue],
            'results': list(self.results),
            'stages_data': []
        }
        for s in self.stages:
            state['stages_data'].append({
                'pair': dict(s.current_pair) if s.current_pair else None,
                'cycles_left': s.cycles_left,
                'p_sum': s.p_sum,
                'p_prod': s.p_prod
            })
        self.history.append(state)

    def load_state(self):
        if not self.history: return
        state = self.history.pop()
        self.tact, self.queue, self.results = state['tact'], state['queue'], state['results']
        for i, data in enumerate(state['stages_data']):
            self.stages[i].current_pair, self.stages[i].cycles_left = data['pair'], data['cycles_left']
            self.stages[i].p_sum, self.stages[i].p_prod = data['p_sum'], data['p_prod']

    def next_tact(self):
        self.save_state()
        self.tact += 1
        for s in self.stages:
            if s.current_pair: s.cycles_left -= 1

        for i in range(self.p - 1, -1, -1):
            s = self.stages[i]
            if s.current_pair and s.cycles_left <= 0:
                if i == self.p - 1:
                    self.results.append(
                        {'id': s.current_pair['id'], 'result': s.p_sum, 'completed_tact': self.tact - 1})
                    s.current_pair = None
                else:
                    next_s = self.stages[i + 1]
                    if next_s.current_pair is None:
                        pair = s.current_pair
                        bit = (pair['a'] >> (i + 1)) & 1
                        prod = (pair['b'] << (i + 1)) if bit else 0
                        next_s.current_pair, next_s.cycles_left = pair, next_s.cycles_required
                        next_s.p_prod, next_s.p_sum = prod, s.p_sum + prod
                        s.current_pair = None

        if self.queue and self.stages[0].current_pair is None:
            pair = self.queue.pop(0)
            s = self.stages[0]
            bit = (pair['a'] >> 0) & 1
            prod = pair['b'] if bit else 0
            s.current_pair, s.cycles_left = pair, s.cycles_required
            s.p_prod, s.p_sum = prod, prod

    def calculate_metrics(self):
        t1 = self.m * sum(self.stage_times)

        tn = self.tact - 1

        if tn > 0:
            self.ky = t1 / tn
            self.eff = self.ky / self.p

    def display(self):
        clear_console()
        print(f"Пример работы арифметического конвейера.")
        print(f"Такт: {self.tact}.")
        print("\nВходная очередь:")
        if not self.queue:
            print("-")
        else:
            for p in self.queue:
                print(f"{p['id']}  {p['a']}={format_bin(p['a'], self.p)}        {p['b']}={format_bin(p['b'], self.p)}")
        print("\n" + "=" * 60)
        for i, s in enumerate(self.stages):
            print(f"Этап {i + 1}.")
            if s.current_pair:
                print(f"Номер пары: {s.current_pair['id']}")
                print(f"Частичная сумма: {format_bin(s.p_sum, 2 * self.p)}    "
                      f"Частичное произведение: {format_bin(s.p_prod, 2 * self.p)}")
            else:
                print("Номер пары: -")
                print("Частичная сумма: -          Частичное произведение: -")
            print()
        print("Результат:")
        if not self.results:
            print("-")
        else:
            for res in self.results:
                print(
                    f"Пара {res['id']}: {res['result']} ({format_bin(res['result'], 2 * self.p)}), завершено на такте {res['completed_tact']}")
        print("\n1) Дальше.  2) Назад.  3) Выход.")


def run_full_simulation(p, queue, times):
    sim = Simulation(p, queue, times)
    while sim.queue or any(s.current_pair for s in sim.stages):
        sim.next_tact()
    sim.calculate_metrics()
    return sim


def plot_metrics(p, times, max_m=20):
    ms = list(range(1, max_m + 1))
    kys = []
    effs = []
    for m in ms:
        queue = [{'id': i + 1, 'a': random.randint(1, 2 ** p - 1), 'b': random.randint(1, 2 ** p - 1)} for i in
                 range(m)]
        sim = run_full_simulation(p, queue, times)
        kys.append(sim.ky)
        effs.append(sim.eff)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(ms, kys, label='Коэффициент ускорения (Ky)')
    plt.xlabel('Количество пар (m)')
    plt.ylabel('Ky')
    plt.title('Коэффициент ускорения')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(ms, effs, label='Эффективность (Eff)')
    plt.xlabel('Количество пар (m)')
    plt.ylabel('Eff')
    plt.title('Эффективность')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    try:
        p = int(input("Введите разрядность p: "))
        m = int(input("Введите количество пар m: "))
        times = [int(input(f"Время в тактах для этапа {i + 1}: ")) for i in range(p)]
    except ValueError:
        return

    input_mode = input("Ввод чисел: random или manual? (r/m): ").strip().lower()
    queue = []
    if input_mode == 'm':
        for i in range(m):
            a = int(input(f"Введите a для пары {i + 1}: "))
            b = int(input(f"Введите b для пары {i + 1}: "))
            queue.append({'id': i + 1, 'a': a, 'b': b})
    elif input_mode == 'r':
        queue = [{'id': i + 1, 'a': random.randint(1, 2 ** p - 1), 'b': random.randint(1, 2 ** p - 1)} for i in
                 range(m)]
    else:
        sys.exit("Неправильный ввод чисел")

    sim = Simulation(p, queue, times)
    while True:
        sim.display()
        choice = input("\n>> ")
        if choice == '1':
            if not sim.queue and not any(s.current_pair for s in sim.stages):
                sim.calculate_metrics()  # Расчет перед выходом
                print("\nВсе пары обработаны!")
                break
            sim.next_tact()
        elif choice == '2':
            sim.load_state()
        elif choice == '3':
            if not sim.queue and not any(s.current_pair for s in sim.stages):
                sim.calculate_metrics()
            else:
                print("Расчет метрик невозможен: работа не завершена.")
            break

    if sim.ky > 0:
        print(f"\nКоэффициент ускорения (Ky): {sim.ky:.2f}")
        print(f"Эффективность (Eff): {sim.eff:.2f}")
        plot_choice = input("Построить графики? (y/n): ").strip().lower()
        if plot_choice == 'y':
            max_m_input = input("Введите максимальное количество пар для графиков (по умолчанию 20): ").strip()
            max_m = int(max_m_input) if max_m_input else 20
            plot_metrics(p, times, max_m)


if __name__ == "__main__":
    main()