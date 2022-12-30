import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


training="C:/Users/shubham kumar/Downloads/archive/train"
testing="C:/Users/shubham kumar/Downloads/archive/test"
validation="C:/Users/shubham kumar/Downloads/archive/valid"
Image_size=[224,224]

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.3,horizontal_flip=True)#This prevent for overfitting
training_set=train_datagen.flow_from_directory(training,target_size=(224,224),batch_size=32,class_mode="categorical")

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.3,horizontal_flip=True)#This prevent for overfitting
training_set=train_datagen.flow_from_directory(training,target_size=(224,224),batch_size=32,class_mode="categorical")

test_datagen=ImageDataGenerator(rescale=1./255)

test_set=test_datagen.flow_from_directory(testing,target_size=(224,224),batch_size=32,class_mode="categorical")

cnn=tf.keras.models.Sequential()

from glob import glob
glob

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu",input_shape=[224, 224, 3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128,activation="relu"))

cnn.add(tf.keras.layers.Dense(units=100,activation="softmax"))

cnn.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

mod=cnn.fit_generator(training_set,validation_data=test_set,epochs=50)

cnn.save("butterfly.h5")

from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np

lab=training_set.class_indices
lab={k:v for v,k in lab.items()}
lab

model1=load_model("butterfly.h5",compile=False)

def output(location):
    img=load_img(location,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model1.predict(img)
    y_class=answer.argmax(axis=-1)
    y=" ".join(str(x) for x in y_class)
    y=int(y)
    res=print("Given butterfly is: ",lab[y])
    return res

img = "C:/Users/shubham kumar/Downloads/archive/train/BLACK HAIRSTREAK/01.jpg"



from PIL import Image
Image.open(img)

output(img)

lab[1]


print("Given butterfly is : ", lab[1])


MODEL_PATH ='butterfly.h5'

# Load your trained model
model = load_model(MODEL_PATH)



def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    y=" ".join(str(z) for z in preds)
    y=int(y)
    res=print("Predicted butterfly is :",lab[y])
    return res
