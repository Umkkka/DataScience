'''
    Инструкция по выполнению:
        1. Загрузите выборку из файла wine.data.
        2. Извлеките из данных признаки и классы. Класс записан в 1-ом столбце, признаки - в столбцах со 2-го по последний.
        3. Оценку качества необходимо провести методом кросс-валидации по 5 блокам. В качестве меры качества используйте
        долю верных ответов.
        4. Найдите точность классификации на кросс-валидации для метода k ближайших соседей при K от 1 до 50.
        При каком k получилось оптимальное качество? Выведите его индекс и значение.
        5. Произведите масштабирование признаков.
        6. Снова найдите оптимальное k на кросс-валидации.
        7. Какое значение k получилось оптимальным после приведения признаков к одному масштабу?
        Выведите его индекс и значение.
'''


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale

data = pd.read_csv(r"C:\Users\Наум\.PyCharm2019.1\Blank\venv\wine.data",
                   names = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                            'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'])

classes = data['class']
del data['class']
evidence = data

score_voc = {}
score_voc_sc = {}

k = 1
while k < 50:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    classific = KNeighborsClassifier(n_neighbors=k) # классификация методо k ближайших соседей
    quality = cross_val_score(classific, X = evidence, y = classes, cv = kf) # кросс-валидация
    quality = np.mean(quality) # усредняем значения полученного массива
    score_voc[k] = quality
    k += 1
max_index = max(score_voc, key = score_voc.get) # индекс значения максимального качества
max_value = score_voc[max_index] # значение максимального качества
print(max_index)
print(round(max_value, 2))

new_evidence = scale(evidence) # масштабирование признаков
k = 1
while k < 50:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    classific = KNeighborsClassifier(n_neighbors=k)
    quality = cross_val_score(classific, X = new_evidence, y = classes, cv = kf)
    quality = np.mean(quality)
    score_voc_sc[k] = quality
    k += 1
max_index_sc = max(score_voc_sc, key = score_voc_sc.get) # индекс значения максимального качества после масштабирования
max_value_sc = score_voc_sc[max_index_sc] # значение максимального качества после масштабирования
print(max_index_sc)
print(round(max_value_sc, 2))

'''
    За счёт приведения признаков к одному масштабу мы получаем оптимальное качество k > 1. Что достаточно логично
    по сравнению с результатом до нормировки. Масштабирование признаков позволяет получить более реальный результат.
'''