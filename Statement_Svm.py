'''
    Вы научитесь:
    - работать с методом опорных векторов (SVM)
    - находить наиболее важные объекты выборки

    Обучающая выборка:
    https://d3c33hcgiwev3.cloudfront.net/_f6284c13db83a3074c2b987f714f24f5_svm-data.csv?Expires=1588636800&Signature=ea0htSsatWG44O6mLWlbLDdI5yMCLZn-IXYcI60nfnQYAbDjDcJ4kgMhl3KNbVRxu2CXTREAT-KwWxgLWgXiASIk3LOJ99HDMRUnZcWFN6MqiTnsfBrnwQBBnfIK7t6181rs3bYXl-MqkrbnDbNXTHzywS8FDYEf109~2jX3IMI_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A

    Инструкция по выполнению:
    1. Загрузите выборку из файла svm-data.csv.
    2. Обучите классификатор с линейным ядром, параметром C = 100000 и random_state=241.
    3. Найдите номера объектов, которые являются опорными.

'''


import numpy as np
from sklearn.svm import SVC

data = np.loadtxt(r"C:\Users\Наум\.PyCharm2019.1\Blank\venv\svm-data.csv", delimiter = ',')

target = data[:, 0]
signs = data[:, 1:]

# обучаем классификатор методом опорных векторов
clf = SVC(random_state=241, kernel='linear', C = 100000)
clf.fit(signs, target)

# вывод опорных объектов
print(clf.support_)