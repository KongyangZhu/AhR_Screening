import keras
from sklearn.model_selection import train_test_split
from keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


def GenerateModel(file, save_path):
    data = file.values
    print('data:\n', data)
    x_data = data[:, 1:-1]
    y_data = data[:, -1].astype(int)
    print(x_data.shape)
    print(y_data.shape)
    print(x_data[1, :].shape)
    print(y_data[:].shape)
    print(type(y_data))

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=3)

    model = keras.Sequential()
    model.add(layers.Dense(units=2048, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=4096, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=2048, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=128, epochs=20)
    # loss, acc = model.evaluate(x_test, y_test, batch_size=128)

    model.save(save_path)
    # print('loss:', loss)
    # print('accuracy:', acc)
    # print('predict:')
    #
    # a = model.predict(x_test)
    # b = a.argmax(axis=1)
    # print(classification_report(y_test, b))
    # print(confusion_matrix(y_test, b))


if __name__ == '__main__':
    fp_file = pd.read_csv(r'Model_FP.csv')
    model_save_path = r'save/ahr_model_dnn.h5'
    GenerateModel(fp_file, model_save_path)
    print('------------------End------------------')


"""
2048+4096+4096+2048
              precision    recall  f1-score   support

           0       0.95      0.98      0.96       713
           1       0.79      0.62      0.69       104

   micro avg       0.93      0.93      0.93       817
   macro avg       0.87      0.80      0.83       817
weighted avg       0.93      0.93      0.93       817

[[696  17]
 [ 40  64]]
"""