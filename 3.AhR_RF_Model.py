import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def GenerateModel(file, model):
    data = file.values
    print('data:\n', data)
    x_data = data[:, 1:-1]
    y_data = data[:, -1].astype(int)
    print(x_data.shape)
    print(y_data.shape)
    print(x_data[1, :].shape)
    print(y_data[:].shape)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=3)

    RF = RandomForestClassifier(n_estimators=250,)
    RF.fit(x_train, y_train)
    joblib.dump(RF, model)


# Save
if __name__ == '__main__':
    file = pd.read_csv(r'Model_FP.csv')
    model = r'save/ahr_model_rf.pkl'
    GenerateModel(file, model)
    print('------------------Done------------------')
