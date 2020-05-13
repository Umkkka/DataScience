'''
    Вы научитесь:
    - работать с методом главных компонент
    - использовать его для вычисления улучшенного индекса Доу-Джонса

    Инструкция по выполнению:
    1. Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 компаний на закрытии торгов за каждый день периода.
    2. На загруженных данных обучите преобразование PCA с числом компоненты равным 10. Скольких компонент хватит, чтобы объяснить 90% дисперсии?
    3. Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.
    4. Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
    5. Какая компания имеет наибольший вес в первой компоненте?

'''


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Подготовка данных - удаляем названия столбцов, удаляем столбец с датами
data = pd.read_csv('close_prices.csv')
data = data.iloc[:,1:]

jonse = pd.read_csv('djia_index.csv')
jonse = jonse.iloc[:,1:]

# Обучаем модель PCA методом с 10 компонентами
pca = PCA(n_components = 10)
pca.fit(data)

# Вычисляем сколько компонент понадобится для объяснения 90% дисперсии
k = 0
sum = 0
for i in range(len(pca.explained_variance_ratio_)):
    k += 1
    value = pca.explained_variance_ratio_[i]
    sum += value
    if sum > 0.9:
        break
print("Need components %d" % k)
print('\n')

# Применяем построенное преобразование к исходным данным
# Выводим первую компоненту
first_component = pd.DataFrame(pca.transform(data)[:, 0])

# Находим корреляцию Пирсона между первой компонентой и индексом Доу-Джонса
corr = np.corrcoef(pd.concat([first_component, jonse], axis=1).T)
print(round(corr[0][1], 2), '\n')

# Находим компанию с наибольшим весом в первой компоненте
i = -1
value = -1
for i in range(len(pca.components_[0])):
    if value < pca.components_[0][i]:
        value = pca.components_[0][i]
        index = i
print(data.columns[index], value)