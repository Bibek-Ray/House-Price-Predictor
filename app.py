from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Housing.csv')
regressor = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    area = int(request.form.get('area'))
    bedroom = int(request.form.get('bedroom'))
    bathroom = int(request.form.get('bathroom'))
    stories = int(request.form.get('stories'))
    mainroad = request.form.get('mainroad')
    guestroom = request.form.get('guestroom')
    basement = request.form.get('basement')
    hotwaterheating = request.form.get('hotwaterheating')
    airconditioning = request.form.get('airconditioning')
    parking = int(request.form.get('parking'))
    prefarea = request.form.get('prefarea')
    furnish = request.form.get('furnish')

    mainroad_value = [0, 1] if mainroad == 'yes' else [1, 0]
    guestroom_value = [0, 1] if guestroom == 'yes' else [1, 0]
    basement_value = [0, 1] if basement == 'yes' else [1, 0]
    hotwaterheating_value = [0, 1] if hotwaterheating == 'yes' else [1, 0]
    airconditioning_value = [0, 1] if airconditioning == 'yes' else [1, 0]
    prefarea_value = [0, 1] if prefarea == 'yes' else [1, 0]

    if furnish == 'furnished':
        furnish_value = [1, 0, 0]
    elif furnish == 'semi-furnished':
        furnish_value = [0, 1, 0]
    else:  # assuming the third option is 'unfurnished'
        furnish_value = [0, 0, 1]

    input_values = [
    area, bedroom, bathroom, stories
    ] + mainroad_value + guestroom_value + basement_value + hotwaterheating_value + airconditioning_value + [
        parking
    ] + prefarea_value + furnish_value

    input_array = np.array(input_values).reshape(1, -1)

    prediction = regressor.predict(input_array)[0]
    print(prediction)
    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
