from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def tune_logistic_regression(X_train, y_train, X_test):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000, solver='liblinear'), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_lr = grid_search.best_estimator_
    y_pred = best_lr.predict(X_test)
    return best_lr, y_pred