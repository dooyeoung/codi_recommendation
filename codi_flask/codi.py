from flask import Flask, render_template, request, jsonify, json
import sqlalchemy 
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, REAL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import DeclarativeMeta
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import pandas as pd
import MySQLdb
from keras.models import load_model 
import requests
import urllib 
import pickle


# html파일 자동 갱신
app = Flask(__name__)
app.config.update(
    TEMPLATES_AUTO_RELOAD = True,
)

 
def init():
    # 패스워드 불러오기
    with open('model/pw.pw', 'rb') as f:
        pw = pickle.load(f) # 단 한줄씩 읽어옴
    
    # load model
    with open("model/qda_goodbad.p", "rb") as f:
        qda = pickle.load(f)

    # db 접속
    db = MySQLdb.connect(
        "127.0.0.1",
        "root",
        pw,
        "codi",
        charset='utf8',
    )  
    model_NN = load_model("model/model_nn.hdf5")
    # 색상 정보 불러오기
    codis_info = pd.read_csv('static/data/codis_info.csv')
    return db, codis_info, model_NN, qda
 
db, codis_info, model_NN, qda = init() 


@app.route("/")
def index():
    return render_template("index.html")
 

@app.route("/recommand/", methods=["POST"])
def recommand():
    color0 = request.values.get("color0")
    color1 = request.values.get("color1")

    color0 = hex_to_rgb(color0)
    color1 = hex_to_rgb(color1)

    results = recommand_color(color0, color1)

    print(results)
    for rcolor in results:
        rcolor = hex_to_rgb(rcolor)
        gb = qda_goodbad(color0, color1, rcolor)
        #print(round(gb * 100, 2))

    color_json = json.loads(json.dumps(results))

    if request.method == 'POST':
        result = {"status":200, "result":color_json}
    else:
        result = {"status":201}
 
    return jsonify(result)


@app.route("/codi/", methods=["POST"])
def codi():
    c1 = hex_to_rgb(request.values.get('color1'))
    c2 = hex_to_rgb(request.values.get('color2'))
    c3 = hex_to_rgb(request.values.get('color3')) 
    cnt = request.values.get('cnt')

    codis = get_similarity(c1, c2, c3, cnt).values[:21] 

    codis = codis_info[codis_info['name'].isin(codis)]


    results = []
    for idx in range(len(codis)):
        row = codis.iloc[idx]
        c1 = tuple(row[['color1_R', 'color1_G', 'color1_B']].values.astype(float))
        c2 = tuple(row[['color2_R', 'color2_G', 'color2_B']].values.astype(float))
        c3 = tuple(row[['color3_R', 'color3_G', 'color3_B']].values.astype(float))
        
        ratio = row[['color1_ratio', 'color2_ratio', 'color3_ratio']].values.tolist()
         
        id_codi = row['name']

        img = get_codi_by_id(id_codi)[0][0]
  
        gb = qda_goodbad(c1, c2, c3)

        results.append([id_codi, img, gb, ratio, (get_hex(c1), get_hex(c2), get_hex(c3))]) 
 
    codi_json = json.loads(json.dumps(results))
    if request.method == 'POST':
        result = {"status":200, "result":codi_json}
    else:
        result = {"status":201}

    return jsonify(result)


 
# functions 
def hex_to_rgb(h): 
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))

# rgb 2 hex
def get_hex(color):
    return '#%02x%02x%02x' % ( int(color[0]), int(color[1]), int(color[2]))

# 코디에서 사용하는 색상 찾기
def get_codi_color(id_codi): 
    SQL_QUERY = """
        SELECT *
        FROM view_codi_color
        where id_codi = {};
    """.format(id_codi)
      
    curs = db.cursor()
    count = curs.execute(SQL_QUERY)
    rows = curs.fetchall()
    return rows

# 코디에서 사용하는 태그 찾기
def get_codi_tag(id_codi):
    SQL_QUERY = """
        SELECT *
        FROM view_codi_tag
        where id_codi = {};
    """.format(id_codi)
      
    curs = db.cursor()
    count = curs.execute(SQL_QUERY)
    rows = curs.fetchall()
    return rows

# 코디검색
def get_codi_by_id(id_codi):
    
    SQL_QUERY = """
        select img from codis
        where codis.id_codi = {} limit 1;
    """.format(id_codi)
      
    curs = db.cursor()
    count = curs.execute(SQL_QUERY)
    rows = curs.fetchall()
    return rows


# 색깔로 코디 찾기
def get_codi_by_colors(colors):
      
    SQL_QUERY = """
        SELECT codis.id_codi, colors.color, codis.img
        from mapping_codi_color
        join codis on codis.id_codi = mapping_codi_color.id_codi
        join colors on colors.id = mapping_codi_color.id_color
        where colors.color in {}
    """.format(colors)
       
    curs = db.cursor()
    count = curs.execute(SQL_QUERY)
    rows = curs.fetchall()
    return rows

# colormind, nn모델사용한 컬러 추천
def recommand_color(color1, color2):
    '''
    recommand 3 colors
    input : main color1(RGB), sub color2(RGB)
    output : 
    recommand color1(RGB, nearest colormind from predict), 
    recommand color2(RGB, predict), 
    recommand color3(RGB, second colormind from predict)
    '''
    input_color = np.hstack([color1, color2]).reshape(1,6)

    # predict color values using dnn

    color_3_pre = model_NN.predict(input_color/256)[0]
    
    # get color values using colormind 
    request_color = [color1, color2, "N"]
    data = {'model' : "default",'input' : request_color}

    li_similar = []
    li_color_3_colormind = []
    
    url = 'http://colormind.io/api/'
    for _ in range(3):
        response = requests.post(url, data = json.dumps(data))
        color_3_colormind = np.array(response.json()["result"][2])
        li_similar.append(np.linalg.norm(color_3_pre - color_3_colormind))
        li_color_3_colormind.append(color_3_colormind)

    li_idx = np.argsort(np.array(li_similar))
    
    c1 = get_hex(li_color_3_colormind[li_idx[0]])
    c2 = get_hex(color_3_pre.astype(int))
    c3 = get_hex(li_color_3_colormind[li_idx[1]])
    
    return c1, c2, c3

# 색상이 무난한지 튀는지 검사
def qda_goodbad(color1, color2, color3):
    '''
    Predict Good/Bad codis using qda
    input :  color1(RGB), color2(RGB), color3(RGB)
    output : good probabilities
    '''
    
    
    
    # change color
    color1 = np.array(color1)
    color2 = np.array(color2)
    color3 = np.array(color3)
    color = np.hstack([color1, color2, color3])
    
    # predict
    prob_good = qda.predict_proba(color.reshape(1, 9))[0]
    
    return prob_good[1]


# 유사도 높은 코디 검색
def get_similarity(c1, c2, c3, cnt):
    if cnt == '1':
        cs = c1
        rc1 = codis_info.filter(['color1_R', 'color1_G', 'color1_B']).values
    elif cnt == '2':
        cs = np.concatenate((c1, c2))
        rc1 = codis_info.filter(['color1_R', 'color1_G', 'color1_B',
                        'color2_R', 'color2_G', 'color2_B']).values
    elif cnt == '3':
        cs = np.concatenate((c1, c2, c3)) 
        rc1 = codis_info.filter(['color1_R', 'color1_G', 'color1_B',
                            'color2_R', 'color2_G', 'color2_B',
                            'color3_R', 'color3_G', 'color3_B']).values
    
    res = np.linalg.norm(rc1 - cs, axis=1)
    
    sort_idx = np.argsort(res)
    res = codis_info['name'][sort_idx]
     
    return res



# 일반 클래스 json 분리
class AlchemyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj.__class__, DeclarativeMeta):
            # an SQLAlchemy class
            fields = {}
            for field in [x for x in dir(obj) if not x.startswith('_') and x != 'metadata']:
                data = obj.__getattribute__(field)
                try:
                    json.dumps(data) # this will fail on non-encodable values, like other classes
                    fields[field] = data
                except TypeError:
                    fields[field] = None
            # a json-encodable dict
            return fields

        return json.JSONEncoder.default(self, obj)

 
if __name__ == "__main__":
    app.run()

# $ gunicorn --reload dss:app
