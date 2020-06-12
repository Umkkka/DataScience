'''
    Вы научитесь:
    - работать с логистической регрессией
    - реализовывать градиентный спуск для ее настройки
    - использовать регуляризацию

    Инструкция по выполнению:
    1. Загрузите данные из файла data-logistic.csv.
    2. Реализуйте градиентный спуск для обычной и L2-регуляризованной логистической регрессии
    3. Запустите градиентный спуск и доведите до сходимости. Рекомендуется ограничить сверху число итераций 10000
    4. Какое значение принимает AUC-ROC на обучении без регуляризации и при ее использовании?

'''


import pandas as pd
import math
from sklearn.metrics import roc_auc_score

import sys
sys.path.append("..")

# импорт данных (цель и признаки)
data = pd.read_csv(r"C:\Users\Наум\.PyCharm2019.1\Blank\venv\data-logistic.csv", header = None)
y = data[0]
x = data.loc[:, 1:]

# определение формул для w1, w2
def fw1(w1, w2, y, x, k, C):
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * x[1][i] * (1.0 - 1.0 / (1.0 + math.exp(-y[i] * (w1 * x[1][i] + w2 * x[2][i]))))
    return w1 + (k * (1.0 / l) * S) - k * C * w1

def fw2(w1, w2, y, x, k, C):
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * x[2][i] * (1.0 - 1.0 / (1.0 + math.exp(-y[i] * (w1 * x[1][i] + w2 * x[2][i]))))
    return w2 + (k * (1.0 / l) * S) - k * C * w2

# реализация градиентного спуска
def grad(y, x, C = 0.0, w1 = 0.0, w2 = 0.0, k = 0.1, err = 0.00001):
    i = 0
    i_max = 10000
    w1_new, w2_new = w1, w2
    while True:
        i += 1
        w1_new, w2_new = fw1(w1, w2, y, x, k, C), fw2(w1, w2, y, x, k, C)
        # вычисление Евклидово пространства
        e = math.sqrt((w1_new - w1) ** 2 + (w2_new - w2) ** 2)
        if i >= i_max or e <= err:
            break
        else:
            w1, w2 = w1_new, w2_new
    return [w1_new, w2_new]

# запускаем градиентный спуск для обычной и L2-регуляризованной логистической регрессий
w1, w2 = grad(y, x)
w1_L2, w2_L2 = grad(y, x, 10.0)

#print(w1, w2)
#print(w1_L2, w2_L2)

# определение сигмоидной функии
def a(x, w1, w2):
    return 1.0 / (1.0 + math.exp(-w1 * x[1] - w2 * x[2]))

# подсчитываем вероятность вышеопределенным алгоритмом
y_score = x.apply(lambda x: a(x, w1, w2), axis=1)
y_score_L2 = x.apply(lambda x: a(x, w1_L2, w2_L2), axis=1)

# вычисляем AUC-ROC на обучении без регуляризации и при ее использовании
auc = roc_auc_score(y, y_score)
auc_L2 = roc_auc_score(y, y_score_L2)

print(round(auc, 3), round(auc_L2, 3))

# Итог: качество классификатора при использовании регуляризации выше, чем без нее
