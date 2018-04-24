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



app = Flask(__name__)
app.config.update(
    TEMPLATES_AUTO_RELOAD = True,
)


models = {}
def init():
    # with open("./models/classification.plk","rb") as f:
    # models["classification"] = pickle.load(f) 

    with open('./model/pw.pw', 'rb') as f:
        pw = pickle.load(f) # 단 한줄씩 읽어옴
    
    db = MySQLdb.connect(
        "127.0.0.1",
        "root",
        pw,
        "codi",
        charset='utf8',
    )  
    engine = sqlalchemy.create_engine("mysql+mysqldb://root:"+pw+"@127.0.0.1/codi")

    codis_info = pd.read_csv('static/data/codis_info.csv')
    return engine, db, codis_info


engine, db, codis_info = init() 
Session = sessionmaker(bind=engine)
session = Session()
 

@app.route("/")
def index():
    return render_template("index.html")
 

#api
@app.route("/color/", methods=["POST"])
def color():
    results = session.query(Colors).all() 

    cinput = hex_to_rgb(request.values.get("color"))
    if request.method == 'POST':
        # json으로 변경하여 데이터 넣기 

        rgb_ls = []
        hex_ls = []
        for color in results:
            rgb_ls.append(np.array(hex_to_rgb(color.color)))
            hex_ls.append(color.color)

        rgb_ls = np.vstack(rgb_ls) 
 
        # 유클리디안 거리 구하기
        uc = np.sqrt(np.sum((rgb_ls - cinput)**2, axis=1))  
        datacom = pd.DataFrame([uc, hex_ls]).T 
        datacom = datacom.sort_values(0) 

        sdata = tuple(datacom[:5][1].values)
  

        codis = get_codi_by_colors(str(sdata))

        # codis = session.query(Codis) \
        # .join(MapColor, MapColor.id_codi == Codis.id_codi) \
        # .join(Colors, MapColor.id_color == Colors.id) \
        # .filter(Colors.color.in_(sdata)).all()

        tags=[]
        rgbs=[]
        for codi in codis:  
            rgbs.append(list(set(get_codi_color(codi[0])))) 
            tags.append(list(set(get_codi_tag(codi[0]))))

        codis = json.loads(json.dumps(codis, cls=AlchemyEncoder)) 
        rgbs = json.loads(json.dumps(rgbs, cls=AlchemyEncoder)) 
        tags = json.loads(json.dumps(tags, cls=AlchemyEncoder)) 


        result = {"status":200, "codis":codis, "rgbs":rgbs, "tags":tags}
    else:
        result = {"status":201}
 
    return jsonify(result)


@app.route("/recommand/", methods=["POST"])
def recommand():
    color0 = request.values.get("color0")
    color1 = request.values.get("color1")

    color0 = hex_to_rgb(color0)
    color1 = hex_to_rgb(color1)

    results = recommand_color(color0, color1)

    print(results)
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

    codis = get_similarity(c1, c2, c3, cnt).values[:20] 

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
        select codis.id_codi, colors.color, codis.img
        from mapping_codi_color
        join codis on codis.id_codi = mapping_codi_color.id_codi
        join colors on colors.id = mapping_codi_color.id_color
        where colors.color in {}
    """.format(colors)
       
    curs = db.cursor()
    count = curs.execute(SQL_QUERY)
    rows = curs.fetchall()
    return rows

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

    li_color_3_recom = []
    # predict color values using dnn
    model = load_model("model/model1_dnn_normal.hdf5")
    color_3_pre = model.predict(input_color/256)[0]
    
    # get color values using colormind 
    request_color = [color1, color2, "N"]
    data = {
    'model' : "default",
    'input' : request_color}

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


def qda_goodbad(color1, color2, color3):
    '''
    Predict Good/Bad codis using qda
    input :  color1(RGB), color2(RGB), color3(RGB)
    output : good probabilities
    '''
    
    # load model
    with open("model/qda_goodbad.p", "rb") as f:
        qda = pickle.load(f)
    
    # change color
    color1 = np.array(color1)
    color2 = np.array(color2)
    color3 = np.array(color3)
    color = np.hstack([color1, color2, color3])
    
    # predict
    prob_good = qda.predict_proba(color.reshape(1, 9))[0]
    
    return prob_good[1]



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

# sql 클래스 정의
Base = declarative_base()
class Codis(Base):
    __tablename__ = 'codis'

    id = Column(Integer, primary_key=True)
    rank = Column(Integer)
    id_codi = Column(Integer)
    img = Column(String)
    link = Column(String)
    tag = Column(String)
    
    def __init__(self, id, rank, id_codi, img, link, tag):
        self.id = id
        self.rank = rank
        self.id_codi = id_codi
        self.img = img
        self.link = link
        self.tag = tag
    
    def __repr__(self):
        return "<Codi {}, {}, {}, {}, {}>".format(self.id, self.rank, self.id_codi, self.img, self.link, self.tag)
    
    
class Colors(Base):
    __tablename__ = 'colors'

    id = Column(Integer, primary_key=True) 
    color = Column(String) 
    
    def __init__(self, id, color):
        self.id = id
        self.color = color 
    
    def __repr__(self):
        return "<Color {}, {}>".format(self.id, self.color)
    

class Tags(Base):
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True)
    tag = Column(String)
    
    def __init__(self, id, tag):
        self.id = id
        self.tag = tag
    
    def __repr__(self):
        return "<Tag {}, {}>".format(self.id, self.tag)
    
    
class Items(Base):
    __tablename__ = 'items'

    id = Column(Integer, primary_key=True) 
    id_item = Column(Integer) 
    id_codi = Column(Integer)
    name = Column(String)
    brand = Column(String)
    price = Column(Integer)
    img = Column(String)
    
    def __init__(self, id, id_item, id_codi, name, brand, price, img):
        self.id = id
        self.id_item = id_item
        self.id_codi = id_codi
        self.name = name
        self.brand = brand
        self.price = price
        self.img = img
    
    def __repr__(self):
        return "<item {}, {}, {}, {}, {}, {}, {}>".format(self.id, self.id_item, self.id_codi, self.name, self.brand, self.price, self.img)
    
    
    
class MapColor(Base):
    __tablename__ = 'mapping_codi_color'
    
    id = Column(Integer, primary_key=True)
    id_codi = Column(Integer)
    ratio = Column(REAL)
    id_color = Column(Integer)
    
    def __init__(id, id_coid, ratio, id_color):
        self.id = id
        self.id_codi = id_coid
        self.ratio = ratio
        self.id_color = id_color
        
    def __repr__(self):
        return "<mapColor {}, {}, {}, {}>".format(self.id, self.id_codi, self.ratio, self.id_color)
    
    
class MapTag(Base):
    __tablename__ = 'mapping_codi_tag'
    
    id = Column(Integer, primary_key=True)
    id_codi = Column(Integer) 
    id_tag = Column(Integer)
    
    def __init__(id, id_coid, id_tag):
        self.id = id
        self.id_codi = id_coid 
        self.id_tag = id_tag
        
    def __repr__(self):
        return "<maptag {}, {}, {}>".format(self.id, self.id_codi, self.id_tag)
    
    
class MapItem(Base):
    __tablename__ = 'mapping_codi_item'
    
    id = Column(Integer, primary_key=True)
    id_codi = Column(Integer) 
    id_item = Column(Integer)
    
    def __init__(id, id_coid, id_item):
        self.id = id
        self.id_codi = id_coid 
        self.id_item = id_item
        
    def __repr__(self):
        return "<mapitem {}, {}, {}>".format(self.id, self.id_codi, self.id_item)
    
 


# $ gunicorn --reload dss:app
