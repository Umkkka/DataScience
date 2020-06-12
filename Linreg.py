'''
     Вы научитесь:
     - использовать линейную регрессию
     - применять линейную регрессию к текстовым данным

     Инструкция по выполнению:
     1. Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла salary-train.csv
     2. Проведите предобработку:
     - Приведите тексты к нижнему регистру
     - Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова
     - Примените TfidfVectorizer для преобразования текстов в векторы признаков.
     Оставьте только те слова, которые встречаются хотя бы в 5 объектах.
     - Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'.
     - Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
     - Объедините все полученные признаки в одну матрицу "объекты-признаки".
     - Обучите гребневую регрессию. Целевая переменная - столбец SalaryNormalized
     - Постройте прогнозы для двух примеров из файла salary-test-mini.csv.
     
'''


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

# Импорт обучающей и тестовой выборок
train = pd.read_csv(r'salary-train.csv')
test = pd.read_csv(r'salary-test-mini.csv')

train_tmp = pd.read_csv(r'salary-train.csv')
test_tmp = pd.read_csv(r'salary-test-mini.csv')

# Заменяем пропущенные значения на специальные строковые величины "nan"
train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

# Заменяем всё, кроме букв и цифр, на пробелы
train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

# Приводим все слова к нижнему регистру
for clf in train.columns[0:1]:
    print(clf)
    train[clf] = train[clf].str.lower()

# Преобразовываем тексты в векторы признаков
# Необходимо оставить только те слова, которые встречаются хотя бы в 5 объектах
vectorizer = TfidfVectorizer(min_df = 5, max_df = 1000000)
X = vectorizer.fit_transform(train['FullDescription'])

# Получаем one-hot кодирование признаков LocationNormalized и ContractTime
enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Объединяем все полученные признаки в одну матрицу "объекты-признаки"
X_train = hstack([X, X_train_categ])
y_train = train_tmp['SalaryNormalized'] # Целевая переменная - SalaryNormalized

# Создание и обучение модели методом гребневой регрессии
clf = Ridge(alpha = 1, random_state = 241)
clf.fit(X_train, y_train)

# Преобразовываем признаки из тестовой выборки по аналогии с признаками обучающей
test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
test['FullDescription'] = test['FullDescription'].str.lower()

# Получаем one-hot кодирование признаков, объединяем признаки в общую матрицу
X_test = vectorizer.transform(test['FullDescription'])
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test, X_test_categ])

# Предсказываем по обученной модели зарплаты из тестовой выборки
y_test = clf.predict(X_test)
print(y_test)
#print(round(56565.3254579, 2), round(37140.63063337, 2))
