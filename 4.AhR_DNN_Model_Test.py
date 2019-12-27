import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


def ModelTest(file, model):
    data = file.values
    x_data = data[:, 1:-1]
    y_data = data[:, -1].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=1)

    model = load_model(model)

    loss, acc = model.evaluate(x_test, y_test)
    print('loss:', loss)
    print('acc:', acc)
    a = model.predict(x_test)
    print(type(a))
    b = a.argmax(axis=1)
    print(classification_report(y_test, b))
    print(confusion_matrix(y_test, b))


if __name__ == '__main__':
    path = r'Model_FP.csv'
    model = r'save/ahr_model_dnn.h5'
    file = pd.read_csv(path)
    ModelTest(file, model)
    print('------------------End------------------')

"""
128+128+64+2:
              precision    recall  f1-score   support

           0       0.94      0.98      0.96       713
           1       0.80      0.57      0.66       104

   micro avg       0.93      0.93      0.93       817
   macro avg       0.87      0.77      0.81       817
weighted avg       0.92      0.93      0.92       817

[[698  15]
 [ 45  59]]
 
1024+256+64+2:
              precision    recall  f1-score   support

           0       0.95      0.98      0.96       713
           1       0.84      0.62      0.72       104

   micro avg       0.94      0.94      0.94       817
   macro avg       0.90      0.80      0.84       817
weighted avg       0.93      0.94      0.93       817

[[701  12]
 [ 39  65]]
 
 2048+4096+4096+2048:
              precision    recall  f1-score   support

           0       0.95      0.98      0.96       713
           1       0.79      0.62      0.69       104

   micro avg       0.93      0.93      0.93       817
   macro avg       0.87      0.80      0.83       817
weighted avg       0.93      0.93      0.93       817

[[696  17]
 [ 40  64]]
"""