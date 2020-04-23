'''
    Инструкция по выполнению:
        1. Загрузите выборку из файла titanic.csv.
        2. Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
        3. Обратите внимание, что признак Sex имеет строковые значения. (необходимо заменить строковые значения для построения дерева)
        4. Выделите целевую переменную — столбец Survived.
        5. В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст. Такие записи при чтении их в pandas принимают значение nan.
         Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
        6. Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию (параметраы конструктора DecisionTreeСlassifier).
        7. Вычислите важности признаков и найдите два признака с наибольшей важностью.
'''

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Набор данных взят с сайта Kaggle: https://www.kaggle.com/c/titanic/data
data = pd.read_csv(r"C:\Users\Наум\.PyCharm2019.1\Blank\venv\titanic.csv", index_col='PassengerId')

data = data[['Pclass','Fare','Sex','Age', 'Survived']]
data = data.dropna() # Удаление строк, имеющих в каком-либо столбце значение nan
purpose = data['Survived'] # Целевая переменная
del data['Survived']

error = np.isnan(data['Age']) # Проверка на то, есть ли значение nan в столбце Age полученной таблицы
if False in error:
    print('Error: in "Age" there is nan')

# Замена строковых значений: мужчины (male) - 1, женщины (female) - 0
data = data.replace('male', 1)
data = data.replace('female', 0)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(data, purpose) # Обучение дерева
importances = clf.feature_importances_
print(importances)