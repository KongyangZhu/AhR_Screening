import pandas as pd
from sklearn.externals import joblib


def AhRPredict(file, model, output):
    data = file.values
    x_data = data[:, 1:]
    print(data)

    # Restore
    RF = joblib.load(model)
    result = RF.predict(x_data)
    print(result.shape)
    a = pd.DataFrame()
    a['CASRN'] = file['CASRN']
    a['result'] = result
    a.to_csv(output, index=0)


if __name__ == '__main__':
    file = pd.read_csv(r'pesticides_FP.csv')
    model = r'save/ahr_model_rf.pkl'
    output = r'pesticidess_AhR_RF_result.csv'
    AhRPredict(file, model, output)
    print('------------------Done------------------')
