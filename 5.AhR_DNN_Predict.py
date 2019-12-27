import pandas as pd
from keras.models import load_model


def AhRPredict(file, output, model):
    data = file.values
    x_data = data[:, 1:]

    model = load_model(model)

    result = model.predict(x_data)
    b = result.argmax(axis=1)
    a = pd.DataFrame()
    a['CASRN'] = file['CASRN']
    a['result'] = b
    a.to_csv(output, index=0)


if __name__ == '__main__':
    model = r'save/ahr_model_dnn.h5'
    path = r'pesticides_FP.csv'
    output = r'pesticides_AhR_DNN_result.csv'

    fp_file = pd.read_csv(path)
    AhRPredict(fp_file, output, model)
    print('------------------End------------------')
