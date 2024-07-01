from modules.data_loader import load_language_data
from modules.feature_processor import preprocess_text, extract_features, extract_features_for_input
from modules.model_trainer import train_naive_bayes, train_logistic_regression
from modules.hyperparameter_analyzer import tune_logistic_regression
from modules.utility import evaluate_model, plot_confusion_matrix
from sklearn.model_selection import train_test_split

def main():
    df = load_language_data('data_files/language_detection.csv')

    if 'Text' not in df.columns:
        raise KeyError("The column 'Text' is not found in the dataset. Please check the column names.")

    df['Text'] = df['Text'].apply(preprocess_text)

    X, y = extract_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb_classifier, y_pred_nb = train_naive_bayes(X_train, y_train, X_test)
    evaluate_model(y_test, y_pred_nb, "Naive Bayes Classifier")
    plot_confusion_matrix(y_test, y_pred_nb, 'Naive Bayes Confusion Matrix', 'output/confusion_matrix_nb.png')
  
    lr_classifier, y_pred_lr = train_logistic_regression(X_train, y_train, X_test)
    evaluate_model(y_test, y_pred_lr, "Logistic Regression Classifier")
    plot_confusion_matrix(y_test, y_pred_lr, 'Logistic Regression Confusion Matrix', 'output/confusion_matrix_lr.png')

    best_lr, y_pred_best_lr = tune_logistic_regression(X_train, y_train, X_test)
    evaluate_model(y_test, y_pred_best_lr, "Best Logistic Regression Classifier")
    plot_confusion_matrix(y_test, y_pred_best_lr, 'Best Logistic Regression Confusion Matrix', 'output/confusion_matrix_best_lr.png')
    
    while True:
        user_input = input("Enter text to detect language (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        predict_language(user_input, nb_classifier, best_lr)

def predict_language(text, nb_model, lr_model):
    processed_text = preprocess_text(text)
    X_input = extract_features_for_input(processed_text)

    nb_prediction = nb_model.predict(X_input)
    lr_prediction = lr_model.predict(X_input)

    print(f"Naive Bayes Prediction: {nb_prediction[0]}")
    print(f"Logistic Regression Prediction: {lr_prediction[0]}")

if __name__ == "__main__":
    main()
