'''
    Вы научитесь:
    - находить оптимальные параметры для метода опорных векторов
    - работать с текстовыми данными

    Инструкция по выполнению:
    1. Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм".
    2. Вычислите TF-IDF-признаки для всех текстов.
    Обратите внимание, что в этом задании мы предлагаем вам вычислить TF-IDF по всем данным.
    3. Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром
    при помощи кросс-валидации по 5 блокам. В качестве меры качества используйте долю верных ответов (accuracy).
    4. Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге.
    5. Найдите 10 слов с наибольшим абсолютным значением веса.

'''


import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV, KFold

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

# Вычисляем TF-IDF-признаки для всех текстов
vectorizer = TfidfVectorizer()

# Преобразовываем обучающую и тестовую выборки
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

# подбираем параметры
grid = {'C': np.power(10.0, np.arange(-5, 6))} # словарь, задающий сетку параметров для перебора
cv = KFold(n_splits=5, shuffle=True, random_state=241)

# классификатор с методом опорных векторов, для которого подбираются значения параметров
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

C_opt = gs.best_params_ # оптимальный параметр С
C_opt = 1.0

# создаем новый классификатор, который обучается на той же выборке, но уже с оптимальным параметром С
clf2 = SVC(kernel='linear', random_state=241, C = C_opt)
clf2.fit(X, y)

# нахождение 10 слов с наибольшим абсолютным значением веса
weights = np.absolute(clf2.coef_.toarray())
max_weights = sorted(zip(weights[0], vectorizer.get_feature_names()))[-10:]
max_weights.sort(key=lambda x: x[1])
print(max_weights)
