import pickle
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def qda_goodbad(color1, color2, color3):
    '''
    Predict Good/Bad codis using qda
    input :  color1(RGB), color2(RGB), color3(RGB)
    output : good probabilities
    '''

    # load model
    with open("qda_goodbad.p", "rb") as f:
        qda = pickle.load(f)

    # change color
    color1 = np.array(color1)
    color2 = np.array(color2)
    color3 = np.array(color3)
    color = np.hstack([color1, color2, color3])

    # predict
    prob_good = qda.predict_proba(color.reshape(1, 9))[0]

    return prob_good[1]
