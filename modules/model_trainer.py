from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def train_naive_bayes(X_train, y_train, X_test):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)
    y_pred = nb_classifier.predict(X_test)
    return nb_classifier, y_pred

def train_logistic_regression(X_train, y_train, X_test):
    lr_classifier = LogisticRegression(max_iter=1000, solver='liblinear')
    lr_classifier.fit(X_train, y_train)
    y_pred = lr_classifier.predict(X_test)
    return lr_classifier, y_pred