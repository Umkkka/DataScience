'''
    Вы научитесь:
    - работать с персептроном — простейшим вариантом линейного классификатора
    - повышать качество линейной модели путем нормализации признаков

    Выборки для обучения и теста соответственно:
    https://d3c33hcgiwev3.cloudfront.net/_3abd237d917280ba0d83bfe6bd49776f_perceptron-train.csv?Expires=1588464000&Signature=VdqMnrKs0OjdSBnviiTnQSFbVtQOANYjrQdKh-dPI6EUeT5IjaRNqcFfYH4nIwkZ5UtpNDSvKR7jKCbZuAPJfKv9JQJeq3RB4-ov~ozA54Mb~TXd96RAImJQCL0uAkCUQPJrVvTT26mOwFB6VZrZZxVxLAls16e7qIAAe~1AEUY_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A
    https://d3c33hcgiwev3.cloudfront.net/_3abd237d917280ba0d83bfe6bd49776f_perceptron-test.csv?Expires=1588464000&Signature=dV0btOqBzR-rdTZzKLhrQkOC7GdcDJ1JnWBz8zw6kcNW~t2wTy2eiSnuWdQp54vPSrLZaB-5esekSBHxk2i9oGVCwFHxeid5oG3irdwcvDnSEwLu0oLMoye0yW883ujfZy0ud9GStcuwFSF9B29TLqKBidi3E3gdHKQ8QLmu9Zs_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A

    Инструкция по выполнению:
    1. Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
    2. Обучите персептрон.
    3. Подсчитайте качество (долю правильно классифицированных объектов) полученного классификатора на тестовой выборке.
    4. Нормализуйте обучающую и тестовую выборку.
    5. Обучите персептрон на новой выборке. Найдите долю правильных ответов на тестовой выборке.
    6. Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее.

'''


import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# в выборках целевая переменная находится в первом столбце, а признаки - во втором и третьем
data_train = np.loadtxt(r"C:\Users\Наум\.PyCharm2019.1\Blank\venv\perceptron-train.csv", delimiter = ',') # обучающая выборка
data_test = np.loadtxt(r"C:\Users\Наум\.PyCharm2019.1\Blank\venv\perceptron-test.csv", delimiter = ',') # тестовая выборка

# подготовка данных: разбиение их на две выборки: целевая переменная и признаки
target_train = data_train[:, 0]
signs_train = data_train[:, 1:]

target_test = data_test[:, 0]
signs_test = data_test[:, 1:]

# обучение нашей модели на данных обучающей выборки - data_train
clf = Perceptron(random_state=241, max_iter=5, tol=None)
clf.fit(signs_train, target_train)

# "скармливаем" нашей модели признаки тестовой выборки
predictions = clf.predict(signs_test)

# подсчет качества классификатора на тестовой выборке: сравниваем предсказание по признакам и целевую переменную
accuracy = accuracy_score(target_test, predictions)
print(accuracy)

scaler = StandardScaler()

# нормализуем признаки обучающей и тестовой выборки
signs_train_scaled = scaler.fit_transform(signs_train)
signs_test_scaled = scaler.transform(signs_test)

# переобучаем персептрон с нормализованными признаками
clf.fit(signs_train_scaled, target_train)

# "скармливаем" переобученной модели нормализованные признаки тестовой выборки
predictions_sc = clf.predict(signs_test_scaled)

# подсчет качества классификатора
accuracy_sc = accuracy_score(target_test, predictions_sc)
print(accuracy_sc)

# находим разность в качестве на тестовой выборке до нормализации признаков и после
print(round(accuracy_sc - accuracy, 3))

# Нормализация данных дает значительный прирост в качестве — оно увеличивается на 19%
