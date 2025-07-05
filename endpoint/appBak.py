from flask import Flask, jsonify, request, Response
import pandas as pd
import tensorflow as tf
from flask_cors import CORS, cross_origin

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "ML jalan"
# Load data rekomendasi dari file CSV atau database
# data = pd.read_csv('Dataset - Capstone Project - Data Kuesioner (2).csv')

# # Preprocessing data jika diperlukan

# # Load model TensorFlow
# model = tf.keras.models.load_model('my_model.h5',compile=False)

# @app.route('/', methods = ['GET'])
# def home():
#     return Response(data.to_json(orient="records"), mimetype='application/json')
# # Route untuk mendapatkan rekomendasi berdasarkan ID item
# @app.route('/recommend/<int:item_id>', methods=['GET'])
# def recommend(item_id):
#     # Ambil data input
#     input_data = data[data['item_id'] == item_id]

#     # Lakukan preprocessing data input jika diperlukan

#     # Lakukan prediksi menggunakan model TensorFlow
#     predictions = model.predict(input_data)

#     # Proses hasil prediksi dan kirim respons
#     top_n = 5  # Jumlah item rekomendasi yang diinginkan
#     recommendations = predictions.argsort()[0][-top_n:][::-1].tolist()

#     # Proses hasil rekomendasi dan kirim respons
#     result = {
#         'item_id': item_id,
#         'recommendations': recommendations
#     }
#     return jsonify(result)

# # Menjalankan Flask server
# if __name__ == '__main__':
#     app.run(debug=True)