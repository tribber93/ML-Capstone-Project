import tensorflow as tf
import numpy as np
from app import app
from config import mysql
from flask import flash, jsonify, request, Response
import pandas as pd
import itertools
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing

def toLowercase(df):
  return [str(lower).lower() for lower in df]

def toList(df):
    return [i.split(', ') for i in df]

def encode_usia(usia,jenis_usia, jumlah_jenis_usia, user = False):
    # Membuat vektor nol dengan panjang yang sama dengan jumlah usia
    encoded_usia = [0] * jumlah_jenis_usia

    for i in usia:  
      # Menentukan indeks usia yang sesuai
      
      usia_index = jenis_usia.index(i)
      
      # Mengatur nilai 1 pada indeks usia yang sesuai
      encoded_usia[usia_index] = 1

    if user:
      encoded_usia[0] = 1
    
    return encoded_usia

def encode_category(category,jenis_kategori, jumlah_kategori):
    # Membuat vektor nol dengan panjang yang sama dengan jumlah kategori
    encoded_category = [0] * jumlah_kategori

    for i in category:  
      # Menentukan indeks kategori yang sesuai
      category_index = jenis_kategori.index(i)
      
      # Mengatur nilai 1 pada indeks kategori yang sesuai
      encoded_category[category_index] = 1
    
    
    return encoded_category

def gen_user_vecs(user_vec, num_items):
    """ given a user vector return:
        user predict maxtrix to match the size of item_vecs """
    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs
    
# Fungsi rekomendasi kuesioner
def pred_kuesioner(y_p, item, items, maxcount=10):
    count = 0
    df_pred = pd.DataFrame(columns=["kuesioner_id", "ratarata_rating", "judul","deskripsi", "rentang_umur", "kategori", "link"])

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        kuesioner_id = item[i, 0].astype(int) - 1
        new_row = pd.Series({
            "kuesioner_id": item[i, 0].astype(int), 
            "ratarata_rating": np.around(item[i, 2].astype(float), 1), 
            "judul": items[kuesioner_id,1],
            "deskripsi": items[kuesioner_id,2],
            "rentang_umur": items[kuesioner_id, 3], 
            "kategori": items[kuesioner_id,4],
            "link": items[kuesioner_id,6]
        })
        df_pred = pd.concat([df_pred, new_row.to_frame().T], ignore_index=True)
    return df_pred

model = tf.keras.models.load_model('my_model.h5',compile=False)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cost_fn)

@app.route('/kuesioner/<int:user_id>/max_item=<int:max>', methods=["GET", "POST"])
def kuesioner(user_id,max=10):
    conn = mysql.connect()

    # Preprocessing
    df_item = pd.read_sql_query("SELECT * FROM kuesioner",conn)
    df = df_item.drop(columns=['jumlah_rating'])
    # return Response(df.to_json(orient="records"), mimetype='application/json')
    
    features = ['judul', 'deskripsi', 'rentang_usia','kategori']
    for feature in features:
        df[feature] = toLowercase(df[feature])

    try:
        df['rentang_usia']=toList(df['rentang_usia'])
        df['kategori']=toList(df['kategori'])
    except:
        print('Sudah diubah')

    # kategori
    list_kategori = df['kategori']
    list_kategori = list(itertools.chain(*list_kategori))

    df_kategori_count = pd.DataFrame(Counter(list_kategori).most_common(), columns=['kategori', 'Jumlah'])

    # Usia pada kuesioner
    list_usia = df['rentang_usia']
    list_usia = list(itertools.chain(*list_usia))

    df_usia_count = pd.DataFrame(Counter(list_usia).most_common(), columns=['rentang_usia', 'Jumlah'])

    jenis_kategori = df_kategori_count['kategori'].values.tolist()
    jumlah_kategori = len(jenis_kategori)

    jenis_usia = df_usia_count['rentang_usia'].values.tolist()
    jumlah_jenis_usia = len(jenis_usia)

    usia_kategori_encoded = []
    for usia in df['rentang_usia']:
        encoded = encode_usia(usia,jenis_usia,jumlah_jenis_usia)
        usia_kategori_encoded.append(encoded)

    kategori_encoded = []
    for kategori in df['kategori']:
        encoded_category = encode_category(kategori, jenis_kategori, jumlah_kategori)
        kategori_encoded.append(encoded_category)
        
    df_temp = pd.DataFrame(columns=df_usia_count['rentang_usia'].values)
    df_kategori = pd.DataFrame(columns=df_kategori_count['kategori'].values)
    df_temp = df_temp.merge(df_kategori, left_index=True, right_index= True)

    usia_array = np.array(usia_kategori_encoded)
    kategori_array = np.array(kategori_encoded)
    for idx, x in enumerate(df_usia_count['rentang_usia']):
        df_temp[x] = usia_array[:,idx]
    for idx2, x2 in enumerate(df_kategori_count['kategori']):
        df_temp[x2] = kategori_array[:,idx2]
        
    i = 0
    item_vecs = df_temp
    if i != 1:
        id_kuesioner = df['kuesioner_id'].values
        item_vecs.insert(0, "kuesioner_id", id_kuesioner, True)
        item_vecs.insert(2, "ratarata_rating", df['ratarata_rating'], True)
        i += 1
        
    # data user
    df_user = pd.read_sql_query(f"SELECT * FROM history_user Where user_id={user_id}",conn)
    
    list_pekerjaan = [  "mahasiswa",
                        "tenaga pendidikan",
                        "wiraswasta",
                        "aparatur/pejabat negara",
                        "tenaga kesehatan",
                        "pertanian/pertenakan",
                        "tidak bekerja",
                        "agama dan kepercayaa",
                    ]
    pekerjaan = preprocessing.LabelEncoder()
    encode_pekerjaan = pekerjaan.fit(list_pekerjaan)
    df_user['pekerjaan'] = encode_pekerjaan.transform(df_user['pekerjaan'])
    
    if df_user['usia'].values[0] >= 15 and df_user['usia'].values[0] < 18:
        df_user['usia'] = ['18-25']
    elif df_user['usia'].values[0] >= 18 and df_user['usia'].values[0] < 26:
        df_user['usia'] = ['18-25']
    elif df_user['usia'].values[0] >= 26 and df_user['usia'].values[0] < 36:
        df_user['usia'] = ['26-35']
    elif df_user['usia'].values[0] >= 36 and df_user['usia'].values[0] < 46:
        df_user['usia'] = ['36-45']
    elif df_user['usia'].values[0] >= 46 and df_user['usia'].values[0] < 56:
        df_user['usia'] = ['46-55']
    else:
            df_user['usia'] = ['umum']
            
    usia = []
    usia.append(encode_usia(df_user['usia'].values,jenis_usia,jumlah_jenis_usia,user=True))
    
    df_usia_user = pd.DataFrame(columns=df_usia_count['rentang_usia'].values)
    usia = np.array(usia)
    # usia
    for idx3, x3 in enumerate(df_usia_count['rentang_usia']):
        df_usia_user[x3] = usia[:,idx3]
    df_temp_user = df_usia_user.merge(df_user, left_index=True, right_index= True).drop(columns=['user_id', 'usia','ratarata_rating'])
    user_vec = df_temp_user
    # return str(df_temp_user.shape)
    # return str(df_user['usia'].values[0])
    # return Response(df_temp_user.to_json(orient="records"), mimetype='application/json')
    
    user_vecs = gen_user_vecs(user_vec.values,len(item_vecs))
    # return type(user_vec)
    scalerItem = StandardScaler()
    scalerItem.fit(item_vecs)

    scalerUser = StandardScaler()
    scalerUser.fit(user_vecs)

    scalerTarget = StandardScaler()
    # scale our user and item vectors
    suser_vecs = scalerUser.transform(user_vecs)
    sitem_vecs = scalerItem.transform(item_vecs)

    # make a prediction
    y_p = model.predict([suser_vecs, sitem_vecs])

    scalerTarget = MinMaxScaler((-1, 1))
    scalerTarget.fit(y_p)
    # unscale y prediction 
    y_pu = scalerTarget.inverse_transform(y_p)

    # sort the results, highest prediction first
    sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
    sorted_ypu   = y_pu[sorted_index]
    sorted_items = item_vecs.values[sorted_index]  #using unscaled vectors for display
    df_pred = pred_kuesioner(sorted_ypu, sorted_items, df.values, maxcount = max)
    return Response(df_pred.to_json(orient="records"), mimetype='application/json')
    # return str(type(item_vecs))

@app.errorhandler(404)
def showMessage(error=None):
    message = {
        'status': 404,
        'message': 'Record not found: ' + request.url,
    }
    respone = jsonify(message)
    respone.status_code = 404
    return respone
    


if __name__ == "__main__":
    app.run(debug=True)