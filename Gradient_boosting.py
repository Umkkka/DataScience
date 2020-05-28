'''
    Вы научитесь:
    - Работать с градиентным бустингом и подбирать его гиперпараметры
    - Сравнивать разные способы построения композиций
    - Понимать, в каком случае лучше использовать случайный лес, а в каком — градиентный бустинг
    - Использовать метрику log-loss

    Инструкция по выполнению:
    1. Загрузите выборку из файла gbm-data.csv с помощью pandas и преобразуйте ее в массив numpy
    2. Разбейте выборку на обучающую и тестовую, используя функцию train_test_split с параметрами test_size = 0.8 и random_state = 241.
    3. Обучите GradientBoostingClassifier с параметрами n_estimators=250, verbose=True, random_state=241.
     Для каждого значения learning_rate из списка [1, 0.5, 0.3, 0.2, 0.1] проделайте следующее:
     - Используйте метод staged_decision_function для предсказания качества на обучающей и тестовой выборке на каждой итерации.
     - Преобразуйте полученное предсказание с помощью сигмоидной функции
     - Вычислите и постройте график значений log-loss на обучающей и тестовой выборках
     - Найдите минимальное значение метрики и номер итерации, на которой оно достигается.
    4. Приведите минимальное значение log-loss на тестовой выборке и номер итерации, на котором оно достигается, при learning_rate = 0.2.
    5. На этих же данных обучите RandomForestClassifier с количеством деревьев, равным количеству итераций, на котором достигается наилучшее качество у градиентного бустинга из предыдущего пункта, c random_state=241 и остальными параметрами по умолчанию. Какое значение log-loss на тесте получается у этого случайного леса?

'''



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss

data = pd.read_csv('gbm-data.csv')

y = data['Activity'].values
X = data.drop('Activity', axis = 1).values
sigmoid_test_arr, sigmoid_train_arr = [], []
test_pred_arr, train_pred_arr = [], []
test_tuples, train_tuples = [], []

# Разбиваем выборку на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 241)

#arr = [1, 0.5, 0.3, 0.2, 0.1]
arr_1 = [0.2]
for i in range(len(arr_1)):
    gbc = GradientBoostingClassifier(n_estimators = 250, learning_rate = arr_1[i], verbose = True, random_state = 241)
    gbc.fit(X_train, y_train)
    # Предсказываем качество модели на обучающей и тестовой выборке
    train_pred = gbc.staged_decision_function(X_train)
    test_pred = gbc.staged_decision_function(X_test)

    # Преобразовываем полученный массив предсказаний с помощью сигмоидной функции
    for i, val in enumerate(train_pred):
        sigmoid = 1 / (1 + np.exp(-val))
        train_pred_arr.append(log_loss(y_train, sigmoid))

    for i, val in enumerate(test_pred):
        sigmoid = 1 / (1 + np.exp(-val))
        test_pred_arr.append(log_loss(y_test, sigmoid))

    # Заполняем новый список значениями метрики на тестовой выборке
    i = 0
    for s in test_pred_arr:
        i += 1
        test_tuples.append((i, s))

    # Сортируем полученный из прошлого действия список для нахождения минимального значения метрики и номера его итерации
    for t in sorted(test_tuples, key=lambda x: x[1]):
        print(t)

    # Строим график значений log-loss на обучающей и тестовой выборках
    plt.plot(test_pred_arr, 'r', linewidth=2)
    plt.plot(train_pred_arr, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()
    # График качества похож на переобучение - overfitting

# Наилучшее качество у градиентного бустинга достигается при итерации под номером 37
# Обучаем модель с количеством соседей равных вышеупомянутому значению
clf = RandomForestClassifier(n_estimators = 37, random_state = 241)
clf.fit(X_train, y_train)

# Находим значение log-loss на тесте
proba = clf.predict_proba(X_test)
loss = log_loss(y_test, proba)
print('Качество на тесте у случайного леса: ',round(loss, 2))
print('Качество на тесте у метода градиентного бустинга: ', round(0.5301645204906471, 2))

# Несмотря на то что в градиентном бустинге гораздо более слабые базовые алгоритмы, он выигрывает у случайного леса благодаря более "направленной" настройке — каждый следующий алгоритм исправляет ошибки имеющейся композиции.
# Также он обучается быстрее случайного леса благодаря использованию неглубоких деревьев.
# В то же время, случайный лес может показать более высокое качество при неограниченных ресурсах — так, он выиграет у градиентного бустинга на наших данных, если увеличить число деревьев до нескольких сотен