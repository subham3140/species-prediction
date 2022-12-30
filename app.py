# request digunakan untuk melakukan penggunaan metode HTTP seperti GET, POST, dll
from flask import Flask, render_template, request
# tensorflow.keras library untuk praproses
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# tensorflow.keras library untuk menggunakan pretrained model
from tensorflow.keras.models import load_model
# untuk perhitungan komputasi
import numpy as np
# untuk regex pada string
import re
# pengaturan direktori
import os
# tanggalan
from datetime import date
import base64
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import glob

lab={0: 'ADONIS',
 1: 'AFRICAN GIANT SWALLOWTAIL',
 2: 'AMERICAN SNOOT',
 3: 'AN 88',
 4: 'APPOLLO',
 5: 'ARCIGERA FLOWER MOTH',
 6: 'ATALA',
 7: 'ATLAS MOTH',
 8: 'BANDED ORANGE HELICONIAN',
 9: 'BANDED PEACOCK',
 10: 'BANDED TIGER MOTH',
 11: 'BECKERS WHITE',
 12: 'BIRD CHERRY ERMINE MOTH',
 13: 'BLACK HAIRSTREAK',
 14: 'BLUE MORPHO',
 15: 'BLUE SPOTTED CROW',
 16: 'BROOKES BIRDWING',
 17: 'BROWN ARGUS',
 18: 'BROWN SIPROETA',
 19: 'CABBAGE WHITE',
 20: 'CAIRNS BIRDWING',
 21: 'CHALK HILL BLUE',
 22: 'CHECQUERED SKIPPER',
 23: 'CHESTNUT',
 24: 'CINNABAR MOTH',
 25: 'CLEARWING MOTH',
 26: 'CLEOPATRA',
 27: 'CLODIUS PARNASSIAN',
 28: 'CLOUDED SULPHUR',
 29: 'COMET MOTH',
 30: 'COMMON BANDED AWL',
 31: 'COMMON WOOD-NYMPH',
 32: 'COPPER TAIL',
 33: 'CRECENT',
 34: 'CRIMSON PATCH',
 35: 'DANAID EGGFLY',
 36: 'EASTERN COMA',
 37: 'EASTERN DAPPLE WHITE',
 38: 'EASTERN PINE ELFIN',
 39: 'ELBOWED PIERROT',
 40: 'EMPEROR GUM MOTH',
 41: 'GARDEN TIGER MOTH',
 42: 'GIANT LEOPARD MOTH',
 43: 'GLITTERING SAPPHIRE',
 44: 'GOLD BANDED',
 45: 'GREAT EGGFLY',
 46: 'GREAT JAY',
 47: 'GREEN CELLED CATTLEHEART',
 48: 'GREEN HAIRSTREAK',
 49: 'GREY HAIRSTREAK',
 50: 'HERCULES MOTH',
 51: 'HUMMING BIRD HAWK MOTH',
 52: 'INDRA SWALLOW',
 53: 'IO MOTH',
 54: 'Iphiclus sister',
 55: 'JULIA',
 56: 'LARGE MARBLE',
 57: 'LUNA MOTH',
 58: 'MADAGASCAN SUNSET MOTH',
 59: 'MALACHITE',
 60: 'MANGROVE SKIPPER',
 61: 'MESTRA',
 62: 'METALMARK',
 63: 'MILBERTS TORTOISESHELL',
 64: 'MONARCH',
 65: 'MOURNING CLOAK',
 66: 'OLEANDER HAWK MOTH',
 67: 'ORANGE OAKLEAF',
 68: 'ORANGE TIP',
 69: 'ORCHARD SWALLOW',
 70: 'PAINTED LADY',
 71: 'PAPER KITE',
 72: 'PEACOCK',
 73: 'PINE WHITE',
 74: 'PIPEVINE SWALLOW',
 75: 'POLYPHEMUS MOTH',
 76: 'POPINJAY',
 77: 'PURPLE HAIRSTREAK',
 78: 'PURPLISH COPPER',
 79: 'QUESTION MARK',
 80: 'RED ADMIRAL',
 81: 'RED CRACKER',
 82: 'RED POSTMAN',
 83: 'RED SPOTTED PURPLE',
 84: 'ROSY MAPLE MOTH',
 85: 'SCARCE SWALLOW',
 86: 'SILVER SPOT SKIPPER',
 87: 'SIXSPOT BURNET MOTH',
 88: 'SLEEPY ORANGE',
 89: 'SOOTYWING',
 90: 'SOUTHERN DOGFACE',
 91: 'STRAITED QUEEN',
 92: 'TROPICAL LEAFWING',
 93: 'TWO BARRED FLASHER',
 94: 'ULYSES',
 95: 'VICEROY',
 96: 'WHITE LINED SPHINX MOTH',
 97: 'WOOD SATYR',
 98: 'YELLOW SWALLOW TAIL',
 99: 'ZEBRA LONG WING'}

app = Flask(__name__, static_url_path='/static')
# direktori model berada
loaded_model = load_model("butterfly.h5")
print('Model ready!!')

UPLOAD_FOLDER = 'static/butterflies'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import base64


# decoding an image from base64 into raw representation
def convertImage(imgData1):
    with open("output.png", 'wb') as output:
        output.write(base64.b64decode(imgData1))

# load image
def load_image(img_path):
    # Praproses data uji
    img = load_img(img_path, target_size=(224,224,3))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    return img_tensor

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():

    # ketika tombol prediksi ditekan maka akan dilakukan sebuah proses konversi berdasarkan yang dituliskan oleh pengguna
    imgData = request.files['butterfly']
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
    for f in files:
     os.remove(f)
    # encode menjadi sebuah output.png
    convertImage(base64.b64encode(imgData.read()))
    # # membaca gambar
    img = load_image("output.png")
    imgData.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(imgData.filename)))
    # melakukan prediksi
    pred = loaded_model.predict(img)
    # konversi respon menjadi string
    preds=np.argmax(pred, axis=1)
    y=" ".join(str(z) for z in preds)
    img = Image.open(imgData.stream)
    with BytesIO() as buf:
            img.save(buf, 'jpeg')
            image_bytes = buf.getvalue()
    encoded_string = base64.b64encode(image_bytes).decode()

    return render_template('index.html',prediction_text= 'Species is {}'.format(lab[int(y)]), image=encoded_string)



if __name__ == "__main__":
    # run the app locally
    app.run(debug=False)
