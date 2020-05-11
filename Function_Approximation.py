'''
     Задача аппроксимации функции.
     Дана функция: f(x) = sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2).
     Необходимо приблизить указанную функцию с помощью многочленов.
     Ответом на задачу являются коэффцициенты многочлена третьей степени, который достаточно точно аппроксимирует
     исходную функцию.

'''


import matplotlib.pyplot as plt
from scipy.linalg import solve
import numpy as np

# Задаем исходную функцию
f = lambda x: np.sin(x / 5.0) * np.exp(x / 10.0) + 5 * np.exp(-x / 2.0)
polynomial_rank = 3

# Создаем матрицу коэффициентов
A = [[1 ** n for n in range(0, polynomial_rank + 1)],
     [4 ** n for n in range(0, polynomial_rank + 1)],
     [10 ** n for n in range(0, polynomial_rank +1)],
     [15 ** n for n in range(0, polynomial_rank + 1)]]
# Создаем свободный вектор (в точках 1, 4, 10, 15 многочлен должен совпадать с исходной функцией
b = [f(1), f(4), f(10), f(15)]

p = solve(A, b) # Решаем систему

g = lambda x, p: p[0] + p[1] * x + p[2] * x ** 2 + p[3] * x ** 3
x = np.arange(1.0, 15.0, 0.25)

plt.plot(x, f(x))
plt.plot(x, g(x, p))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()

for i in range(4):
     print(round(p[i], 2))