import tensorflow as tf
from tensorflow import keras

print("TensorFlow version: ", tf.__version__)
print("Keras version: ", keras.__version__)

import numpy as np
import pandas as pd
import joblib
import os
from flask import Flask, render_template

from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRING_FIELD = StringField('max_wind_speed', validators=[DataRequired()])

np.random.seed(42)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

class LabForm(FlaskForm):
    longitude = StringField('longitude(1-7)', validators=[DataRequired()])
    latitude = StringField('latitude(1-7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()])
    day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        new_data = pd.DataFrame([{
                "longitude": float(form.longitude.data),
                "latitude": float(form.latitude.data),
                "month": form.month.data,
                "day": form.day.data,
                "avg_temp": float(form.avg_temp.data),
                "max_temp": float(form.max_temp.data),
                "max_wind_speed": float(form.max_wind_speed.data),
                "avg_wind": float(form.avg_wind.data)
            }])
        """X_test = np.array([[float(form.longitude.data),
                            float(form.latitude.data),
                            float(form.month.data),
                            float(form.day.data),
                            float(form.avg_temp.data),
                            float(form.max_temp.data),
                            float(form.max_wind_speed.data),
                            float(form.avg_wind.data)]])
        print(X_test.shape)
        print(X_test)

        data = pd.read_csv('./sanbul2district-divby100.csv', sep=',')

        X = data.values[:, 0:8]
        y = data.values[:, 8]

        scaler = StandardScaler()
        scaler.fit()

        X_test = scaler.transform(X_test)"""

        #model = keras.models.load_model('fires_model.keras')
        #pipeline = joblib.load('pipeline.pkl')

        model = keras.models.load_model(os.path.join(BASE_DIR, 'fires_model.keras'))
        pipeline = joblib.load(os.path.join(BASE_DIR, 'pipeline.pkl'))

        X_test = pipeline.transform(new_data)
        prediction = model.predict(X_test)
        res = np.expm1(prediction[0][0])
        #res = (float)(np.round(res * 100))
        res = np.round(res, 2)

        return render_template('result.html', res=res)
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run()