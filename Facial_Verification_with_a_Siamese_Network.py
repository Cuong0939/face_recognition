# import standard dependencies
import cv2
import os
import random
import numpy as np 
import matplotlib.pyplot as plt
import uuid

from validators import Max

# import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu,True) 

#setup paths
POS_PATH = "data/positive/"
NEG_PATH = "data/negative/"
ANC_PATH = "data/anchor/"


# Move LFW Images to the following repository data/negative
for directory in os.listdir('lfw/'):
  for file in os.listdir(os.path.join('lfw/',directory)):
    EX_PATH = os.path.join("lfw/",directory+"/",file)
    NEW_PATH = os.path.join(NEG_PATH,file)
    os.replace(EX_PATH,NEW_PATH)



# import uuid

cap = cv2.VideoCapture(0)
while cap.isOpened():
  ret,frame = cap.read()
  #Cut down frame to 250x250px
  frame = frame[150:150+250,200:200+250,:]


  #Collect anchors
  if cv2.waitKey(1) & 0xFF == ord('a'):
    imagePath = os.path.join(ANC_PATH,f"{uuid.uuid1()}.jpg")
    cv2.imwrite(imagePath,frame)
  #Collect positives

  if cv2.waitKey(1) & 0xFF == ord('p'):
    imagePath = os.path.join(POS_PATH,f"{uuid.uuid1()}.jpg")
    cv2.imwrite(imagePath,frame)
  cv2.imshow("Image Colection",frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    image = frame
    break

cap.release()
cv2.destroyAllWindows()

##2.x NEW - Data Augmentation
def data_aug(img):
  data = []
  for i in range(9):
    img = tf.image.stateless_random_brightness(img,max_delta=0.02,seed=(1,2))
    img = tf.image.stateless_random_contrast(img,lower=0.6,upper = 1,seed=(1,3))
    img = tf.image.stateless_random_flip_left_right(img,seed=(np.random.randint(100),np.random.randint(100)))
    img = tf.image.stateless_random_jpeg_quality(img,min_jpeg_quality=90,max_jpeg_quality=100,seed=(np.random.randint(100),np.random.randint(100)))
    img = tf.image.stateless_random_saturation(img,lower=0.9,upper=1,seed=(np.random.randint(100),np.random.randint(100)))

    data.append(img)

  return data

for file_name in os.listdir(os.path.join(POS_PATH)):
  img_path = os.path.join(POS_PATH,file_name)
  img =cv2.imread(img_path)
  augemented_images = data_aug(img)

  for image in augemented_images:
    cv2.imwrite(os.path.join(POS_PATH,f'{uuid.uuid1()}.jpg'),image.numpy())




#Get Image Directories
anchor = tf.data.Dataset.list_files(ANC_PATH+"*.jpg").take(300)
positive = tf.data.Dataset.list_files(POS_PATH+"*.jpg").take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+"*.jpg").take(300)


# dir_text = anchor.as_numpy_iterator()
# print(dir_text.next())

#Preprocessing - Scale and Resize
def preprocess(file_path):
  #Read in image from file path
  byte_img = tf.io.read_file(file_path)
  #load in the image
  img =tf.io.decode_jpeg(byte_img)

  # Preprocessing steps - resizing the image to be  100x100x3
  img = tf.image.resize(img,(100,100))
  # Scale image to be between 0 and 1
  img = img / 255.0

  return img

# plt.imshow(preprocess(dir_text.next()))
# plt.show()

#Create Labelled Dataset
# (anchor, positive) => 1,1,1,1,1
# (anchor, negative) => 0,0,0,0,0

positives = tf.data.Dataset.zip((anchor,positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negative = tf.data.Dataset.zip((anchor,negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negative)



# Build Train and Test Partition
def preprocess_twin(input_img,validation_img,label):
  return (preprocess(input_img),preprocess(validation_img),label)

# res = preprocess_twin(*examples)
# plt.imshow(res[1])
# plt.show()

## Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data =data.shuffle(buffer_size=1000)

# Training partition
train_data = data.take(round(len(data) * .7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(round(len(data) * .7))
test_data = data.take(round(len(data) * .3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# print(train_data)
# print(test_data)

train_sample = train_data.as_numpy_iterator()
train_sample = train_sample.next()
# print(train_sample)

##Model Engineering
#Build Embedding Layer


def make_embedding():
  inp = Input(shape=(100,100,3),name='input_image')

  #First block
  c1 = Conv2D(64,(10,10),activation='relu')(inp)
  m1 = MaxPooling2D(64,(2,2),padding='same')(c1)

  #Second block
  c2 = Conv2D(128,(7,7),activation='relu')(m1)
  m2 = MaxPooling2D(64,(2,2),padding='same')(c2)

  # Third block
  c3 = Conv2D(256,(4,4),activation='relu')(m2)
  m3 = MaxPooling2D(64,(2,2),padding='same')(c3)

  # Final embedding block
  c4 = Conv2D(256,(4,4),activation='relu')(m3)
  f1 = Flatten()(c4)
  d1 = Dense(4096,activation='sigmoid')(f1)

  return Model(inputs=[inp],outputs=[d1],name='embedding')

embedding = make_embedding()

# print(embedding.summary())


# Build distance layer
#Siamese L1 Distance class
class L1Dist(Layer):
  #Init method - inheritance
  def __init__(self,**kwangs):
      super().__init__()

  #Magic happens here - similarity calculation
  def call(self,input_embedding,validation_embedding):
    return tf.math.abs(input_embedding - validation_embedding)


#Make Siamese model

def make_siamese_model():
  #Anchor image input in the network
  input_image =Input(name='input_img',shape=(100,100,3))

  #Validation image in the network
  validation_image = Input(name='validation_img',shape=(100,100,3))

  #Combine siamese distance components
  siamese_layer = L1Dist()
  siamese_layer._name = 'distance'
  distances = siamese_layer(embedding(input_image),embedding(validation_image))

  #Classification Layer
  classifier = Dense(1,activation='sigmoid')(distances)

  return  Model(inputs=[input_image,validation_image],outputs=classifier,name='SiameseNetwork')

siamese_model = make_siamese_model()
# print(siamese_model.summary())
## TRAINING
# Setup loss and optimizer
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)
# Establish checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt,siamese_model = siamese_model)
# Build train step function
# print(train_data.as_numpy_iterator().next())
@tf.function
def train_step(batch):

  # Record all of our operations
  with tf.GradientTape() as Tape:
    # Get anchor and positive/negative image
    X = batch[:2]
    # Get label
    y = batch[2]

    # Forward pass
    yhat = siamese_model(X,training=True)
    #Calculate loss
    loss = binary_cross_loss(y,yhat)
  print(loss)

  #Calculate gradients
  grad = Tape.gradient(loss,siamese_model.trainable_variables)
  #Calculate updated weight and apply to siamese model
  opt.apply_gradients(zip(grad,siamese_model.trainable_variables))

  #return loss
  return loss

# Build training loop
def train(data,epochs):
  #loop through epochs
  for epoch in range(1,epochs+1):
    print(f'\n Epoch {epoch}/{epochs}')
    progbar = tf.keras.utils.Progbar(len(data))

    # Creating a metric  object
    r = Recall()
    p = Precision()
    #Loop through each batch
    for idx,batch in enumerate(data):
      #run trainstep here
      loss = train_step(batch)
      yhat = siamese_model.predict(batch[:2])
      r.update_state(batch[2],yhat)
      p.update_state(batch[2],yhat)
      progbar.update(idx+1)
    print(loss.numpy(),r.result().numpy(),p.result().numpy())

    # save checkpoints
    if epoch %10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
# train the model
EPOCH = 50
train(train_data,EPOCH)
# evaluate model
#import metrics
#import metric calculations
from tensorflow.keras.metrics import Precision,Recall

## Make Prediction
test_input,test_val,y_true = test_data.as_numpy_iterator().next()

y_hat = siamese_model.predict([test_input,test_val])
[1 if prediction > 0.5 else 0 for prediction in y_hat]
print(y_true)

# #Calculating Recall value
# m=Recall()
# m.update_state(y_true,y_hat)
# print(m.result().numpy())

# #Calculating Precision value
# m=Precision()
# m.update_state(y_true,y_hat)
# print(m.result().numpy())


r=Recall()
p=Precision()

for test_input,test_val,y_true in test_data.as_numpy_iterator():
  yhat = siamese_model.predict([test_input,test_val])
  r.update_state(y_true,yhat)
  p.update_state(y_true,yhat)

print(r.result().numpy(),p.result().numpy())



# Set plot size 
plt.figure(figsize=(10,8))

# Set first subplot
plt.subplot(1,2,1)
plt.imshow(test_input[0])

# Set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[0])

# Renders cleanly
plt.show()

#Save model
siamese_model.save('siamesemodelv2.h5')
# loadmodel
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5',custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# Make predictions with reloaded model
siamese_model.predict([test_input, test_val])

# View model summary
siamese_model.summary()

os.listdir(os.path.join('application_data', 'verification_images'))
os.path.join('application_data', 'input_image', 'input_image.jpg')


## Verfication function
def verify(model,dectection_threshold,verification_threshold):
  results = []
  for image in os.listdir(os.path.join('application_data/','verification_images/')):
    input_img = preprocess(os.path.join('application_data/','input_image/','input_image.jpg'))
    validation_img = preprocess(os.path.join('application_data/','verification_images/',image))

    #make preditions
    result = model.predict(list(np.expand_dims([input_img,validation_img],axis = 1)))
    results.append(result)
  # Detection Threshold: Metric above which a prediciton is considered positive
  detection = np.sum(np.array(results) > dectection_threshold)

  #Verification threshold: Proportion of positive predictions/total positive samples
  verification = detection / len(os.listdir(os.path.join('application_data/','verification_images/')))
  verified = verification > verification_threshold

  return results, verified


##OpenCV Real Time Verification

cap = cv2.VideoCapture(0)
while cap.isOpened():
  ret,frame = cap.read()
  frame = frame[150:150+250,200:200+250,:]
  cv2.imshow('Verification',frame)

  # verification trigger
  if cv2.waitKey(10) & 0xFF == ord('v'):

    cv2.imwrite(os.path.join('application_data/','input_image/','input_image.jpg'),frame)
    #Run verfication
    results,verified = verify(siamese_model,0.5,0.5)
    print(verified)

  if cv2.waitKey(10) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()

np.sum(np.squeeze(results) > 0.9)
print(results)