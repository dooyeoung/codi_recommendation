from flask import Flask, render_template, request, jsonify
import pickle
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, REAL
from sqlalchemy.orm import sessionmaker
import sqlalchemy
import json
from sqlalchemy.ext.declarative import DeclarativeMeta
import numpy as np
import pandas as pd
import MySQLdb

app = Flask(__name__)
app.config.update(
    TEMPLATES_AUTO_RELOAD = True,
)


models = {}
def init():
    # with open("./models/classification.plk","rb") as f:
    # models["classification"] = pickle.load(f) 

    with open('pw.pw', 'rb') as f:
    	pw = pickle.load(f) # 단 한줄씩 읽어옴
    
    db = MySQLdb.connect(
    "127.0.0.1",
    "root",
    pw,
    "codi",
    charset='utf8',
    ) 

    engine = sqlalchemy.create_engine("mysql+mysqldb://root:"+pw+"@127.0.0.1/codi")
    return engine, db

engine, db = init() 
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
 
        uc = np.sqrt(np.sum((rgb_ls - cinput)**2, axis=1))  
        datacom = pd.DataFrame([uc, hex_ls]).T 
        datacom = datacom.sort_values(0) 

        sdata = datacom[:5][1].values 
  

        codis = session.query(Codis) \
        .join(MapColor, MapColor.id_codi == Codis.id_codi) \
        .join(Colors, MapColor.id_color == Colors.id) \
        .filter(Colors.color.in_(sdata)).all()

        tags=[]
        rgbs=[]
        for codi in codis:  
            rgbs.append(list(set(get_codi_color(codi.id_codi)))) 
            tags.append(list(set(get_codi_tag(codi.id_codi))))

        codis = json.loads(json.dumps(codis, cls=AlchemyEncoder)) 
        rgbs = json.loads(json.dumps(rgbs, cls=AlchemyEncoder)) 
        tags = json.loads(json.dumps(tags, cls=AlchemyEncoder)) 


        result = {"status":200, "codis":codis, "rgbs":rgbs, "tags":tags}
    else:
        result = {"status":201}
 
    return jsonify(result)

 
# functions 
def hex_to_rgb(h): 
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))


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
