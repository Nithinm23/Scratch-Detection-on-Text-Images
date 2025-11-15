import cv2, numpy as np
from model import build_model

def predict(model_path, image_path, thr=0.5):

    model = keras.models.load_model(model_path, compile=False)

    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (256,256)) / 255.0

    out = model.predict(np.expand_dims(rgb,0), verbose=0)
    prob = float(out[0][0][0])
    mask = out[2][0,...,0]

    label = "BAD" if prob > thr else "GOOD"
    return prob, label, mask
