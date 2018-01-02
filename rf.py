import os

from flask import Flask, jsonify

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route('/', methods=['GET'])
def test():
    return jsonify({'message': 'Hello World!'})


@app.route('/detect/<string:name>', methods=['GET'])
def HeartAttackCal(name):
    s = name
    # s= "58,4,3,3,3,3,2,2,2,2,2,3,3,3,3,3,3,3,4,4,5,2,3,3,3,3,3,3,3,3,2,3,3,5,6,2,5,5,2"
    input_String = [s.split(",")]

    Ami_data = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"), delimiter=',', encoding="utf-8-sig")
    Ami_data = Ami_data.sample(frac=1).reset_index(drop=True)

    features = Ami_data.columns[2:40]
    # length = Ami_data.shape[0]

    input_Data = pd.DataFrame(input_String, columns=features)

    x_Data_frame = Ami_data.iloc[:, 2:40]
    x_data = Ami_data[list(x_Data_frame)].values

    y_Data_frame = Ami_data.iloc[:, 40:43]
    y_data = Ami_data[list(y_Data_frame)].values

    y_test = y_data[222:]

    Ami_data['Result'] = Ami_data[list(y_Data_frame)].idxmax(axis=1)

    train_Data = Ami_data[:221]
    # test_Data = Ami_data[310:]
    test_Data = input_Data

    factorized_Value = pd.factorize(train_Data['Result'])[0]

    clf = RandomForestClassifier()
    clf.fit(train_Data[features], factorized_Value)

    # clf.score(test_Data[features],factorized_Value) # Accuracy

    output = Ami_data.Result[clf.predict(test_Data[features])]

    return jsonify({
        "result": list(output.to_dict().values())[0]
    })


if __name__ == '__main__':
    app.run(debug=True, port=8080)
