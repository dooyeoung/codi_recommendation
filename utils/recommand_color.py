from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import requests
import json
import pandas as pd
import numpy as np

def recommand_color(color1, color2):
    '''
    recommand 3 colors
    input : main color1(RGB), sub color2(RGB)
    output : recommand color1(RGB, nearest colormind from predict), recommand color2(RGB, predict), recommand color3(RGB, second colormind from predict)
    '''
    input_color = np.hstack([color1, color2]).reshape(1,6)

    li_color_3_recom = []
    # predict color values using dnn
    model = load_model("model1_dnn_normal.hdf5")
    color_3_pre = model.predict(input_color/256)[0]

    # get color values using colormind
    li_color_3_colormind = []
    request_color = [color1, color2, "N"]
    data = {
    'model' : "default",
    'input' : request_color}

    li_similar = []
    li_color_3_colormind = []

    for _ in range(3):
        response = requests.post(url, data = json.dumps(data))
        color_3_colormind = np.array(response.json()["result"][2])
        li_similar.append(np.linalg.norm(color_3_pre - color_3_colormind))
        li_color_3_colormind.append(color_3_colormind)

    li_idx = np.argsort(np.array(li_similar))
    # recommend color1
    li_color_3_recom.append(li_color_3_colormind[li_idx[0]])
    # recommend color2
    li_color_3_recom.append(color_3_pre)
    # recommend color3
    li_color_3_recom.append(li_color_3_colormind[li_idx[1]])

    return(li_color_3_recom)

# predict = [color1[idx], color2[idx], color_3_pre]
# bar_predict = plot_colors([0.33, 0.33, 0.33], predict, h=50, w=300)
