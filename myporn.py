import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy import sparse
def features(df):
    words_ru = ['порн', 'очко', 'ебет', 'раком', 'секс', 'лесби',
             'куни', 'голы', 'эроти', 'раздетый', 'соски',
             'обнаженн', 'интим', 'видео', 'гей', 'орал', 'анал', 'еб', 'трах', 'жар', 'стринг','жоп','конч',
                'вагин', 'попа', 'попу', 'попк', 'пизд', 'орги', 'фистинг', 'сперм']
    words_en = ["porn", "sex", "x", "xxx", "lesbi", "fuck",  "horny", "naked", "dirty"
               "gay", "vagin", "oral"
               "nude", "tit", "dick", "girl", "hot", "undress", "anal", "slut", "topless", "pussy", 
                "video", "ero", "ass", "cum", "sperm"]
             
    title = df["title"].values
    url = df["url"].values
    
    result = []
    for i, val in enumerate(title):
        result.append([])
        for j in words_ru:
            result[-1].append(int(j in val.lower()))
        for j in words_en:
            result[-1].append(int(j in val.lower()))
        for j in words_en:
            result[-1].append(int(j in url[i].lower()))
    return result


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
X_train = train_df["title"].values + ' ' + train_df["url"].values
X_test = test_df["title"].values+ ' '+test_df["url"].values
y_train = train_df["target"].astype(int).values
vectorizer = TfidfVectorizer()
model = RandomForestClassifier(n_estimators = 200)
X_train_vectorized = sparse.hstack((vectorizer.fit_transform(X_train), sparse.csr_matrix(np.asarray(features(train_df)))))
model.fit(
    X_train_vectorized,
    y_train
)
y_pred = model.predict(
    X_train_vectorized
)
print(f1_score(y_train, y_pred))
X_test_vectorized = sparse.hstack((vectorizer.transform(X_test), sparse.csr_matrix(np.asarray(features(test_df)))))

test_df["target"] = model.predict(X_test_vectorized).astype(bool)

test_df[["id", "target"]].to_csv("my_porn_test.csv", index=False)
