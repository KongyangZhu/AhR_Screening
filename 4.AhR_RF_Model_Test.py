import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib


def ModelTest(file, model):
    data = file.values
    x_data = data[:, 1:-1]
    y_data = data[:, -1].astype(int)
    print(type(x_data))

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=3)

    # Restore
    RF = joblib.load(model)
    RF.score(x_test, y_test)
    print(classification_report(y_test, RF.predict(x_test)))
    print(confusion_matrix(y_test, RF.predict(x_test)))


if __name__ == '__main__':
    file = pd.read_csv(r'Model_FP.csv')
    model = r'save/ahr_model_rf.pkl'
    ModelTest(file, model)
    print('------------------Done------------------')

"""
random_state = 1 :
              precision    recall  f1-score   support
           0       0.93      0.98      0.95       713
           1       0.80      0.46      0.59       104
   micro avg       0.92      0.92      0.92       817
   macro avg       0.86      0.72      0.77       817
weighted avg       0.91      0.92      0.91       817
[[701  12]
 [ 56  48]]

random_state = 3 :
              precision    recall  f1-score   support
           0       0.93      0.99      0.96       713
           1       0.86      0.47      0.61       104
   micro avg       0.92      0.92      0.92       817
   macro avg       0.89      0.73      0.78       817
weighted avg       0.92      0.92      0.91       817
[[705   8]
 [ 55  49]]
"""