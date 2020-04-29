'''
    Вы научитесь подбирать конкретную метрику Минковского для задачи

    Мы будем использовать в данном задании набор данных Boston, где нужно предсказать стоимость жилья на основе различных
    характеристик расположения (загрязненность воздуха, близость к дорогам и т.д.).
    Подробнее о признаках можно почитать по адресу https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

    Инструкция по выполнению:
    1. Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
    2. Необходимо перебрать разные варианты метрики p по сетке от 1 до 10 так, чтобы было протестировано 200 вариантов.
    Используйте KNeighborsRegressor с параметром weights=’distance’. Данный параметр добавляет в алгоритм веса, зависящие
    от расстояния до ближайших соседей.
    3. В качестве метрики качества необходимо использовать среднеквадратичную ошибку.
    4. Необходимо оценить качество с помощью кросс-валидации по 5 блокам.
    5. Определите, при каком p качество на кросс-валидации оказалось оптимальным.

'''


import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale

Boston = load_boston()

purpose = Boston['target'] # целевой вектор
signs = Boston['data'] # признаки
signs = scale(signs) # приводим признаки к одному масштабу

score_voc = {}

k = 1
while k < 10:
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    classific = KNeighborsRegressor(n_neighbors = 5, weights = 'distance', p = k)
    quality = cross_val_score(classific, X = signs, y = purpose, cv = kf, scoring = 'neg_mean_squared_error', )
    quality = np.mean(quality) # усредняем значения полученного массива
    score_voc[k] = quality
    k += 0.05

max_p = max(score_voc, key = score_voc.get)
print(max_p)