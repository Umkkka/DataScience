'''
    Вы научитесь:
    - Вычислять различные меры качества классификации: долю правильных ответов, точность, полноту, AUC-ROC и т.д.
    - Сравнивать алгоритмы классификации при наличии ограничений на точность или полноту

    Инструкция по выполнению:
    1. Загрузите файл classification.csv.
    В нем записаны истинные классы объектов выборки (true) и ответы некоторого классификатора (pred).
    2. Заполните таблицу ошибок классификации. Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям.
    3. Посчитайте основные метрики качества классификатора: доля верно угаданных, точность, полнота, f-мера.
    4. Имеется четыре обученных классификатора.
    В файле scores.csv записаны истинные классы и значения степени принадлежности положительному классу для каждого классификатора на некоторой выборке.
    Посчитайте площадь под ROC-кривой для каждого классификатора. Какой классификатор имеет наибольшее значение метрики AUC-ROC?
    5. Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?

'''


import pandas as pd
import numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

# импорт данных
data = pd.read_csv(r"C:\Users\Наум\.PyCharm2019.1\Blank\venv\classification.csv", header = None)
true = data[0]
pred = data.loc[:, 1]

TP = 0 # True Positive (1-1)
FP = 0 # False Positive (1-0)
FN = 0 # False Negative (0-1)
TN = 0 # True Negative (0-0)

i = 0
for i in range(len(true)):
    if true[i] == '1' and pred[i] == '1':
        TP += 1
    if true[i] == '0' and pred[i] == '1':
        FP += 1
    if true[i] == '1' and pred[i] == '0':
        FN += 1
    if true[i] == '0' and pred[i] == '0':
        TN += 1

print(TP, FP, FN, TN, '\n')

data = pd.read_csv(r"C:\Users\Наум\.PyCharm2019.1\Blank\venv\classification.csv")

print('accuracy score =', round(accuracy_score(data.true, data.pred), 2))
print('precision score =', round(precision_score(data.true, data.pred), 2))
print('recall score =', round(recall_score(data.true, data.pred), 2))
print('f1 score =', round(f1_score(data.true, data.pred), 2))
print('\n')

scores = pd.read_csv(r"C:\Users\Наум\.PyCharm2019.1\Blank\venv\scores.csv")

print('auc-roc_logreg =', round(roc_auc_score(scores.true, scores.score_logreg),2))
print('auc-roc_svm =', round(roc_auc_score(scores.true, scores.score_svm), 2))
print('auc-roc_knn =', round(roc_auc_score(scores.true, scores.score_knn), 2))
print('auc-roc_tree =', round(roc_auc_score(scores.true, scores.score_tree), 2))
print('\n')

curve = {}
for score in scores.keys()[1:]:
    df = pd.DataFrame(columns=('precision', 'recall'))
    df.precision, df.recall, thresholds = precision_recall_curve(scores.true, scores[score])
    curve.update({score: df[df['recall'] >= 0.7]['precision'].max()})
print(curve)
best_model = max(curve, key=curve.get)
print(best_model)
