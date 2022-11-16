### Hola, I'm Pritish! 👋


- 🔭 I’m currently working on virtual internship programs.
- 🌱 I’m currently learning new technologies.
- 💬 Ask me about business insights.
- 📫 How to reach me: 
-                      Twitter - @pritishkshetre
-                      Instagram - @pritishkshetre
-                      Facebook - @pritishkshetre
-                      Linked In - @pritish-kshetre
2)IMPLEMENTING FEEDFORWARD NEURAL NETWORK WITH KERAS AND TENSORFLOW
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape
x_test.shape
y_train
plt.subplot(221)
plt.title(y_train[0])
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.title(y_train[1])
plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.title(y_train[2])
plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.title(y_train[3])
plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
num_pixels = x_train.shape[1] * x_train.shape[2]
num_pixels
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype(float)
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype(float)
x_train.shape
x_test.shape
x_train = x_train/225
x_test = x_test/225
x_train
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train.shape
def create_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, 
                    activation= 'relu' ))
    model.add(Dense(10, activation= 'softmax' ))
# Compile model
model.compile(loss= 'categorical_crossentropy', optimizer= SGD(), metrics=[ 'accuracy' ])
return model
model=create_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size = 200)
model.evaluate(x_test, y_test, batch_size=1)
model.summary()
new= x_test[45]
new=new.reshape(1,-1)
x_train.shape
new.shape
np.argmax(model.predict(new))
y_test[45]
history= model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=10,batch_size=200)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train', 'val'])



6)OBJECT DETECTION USING TRASFER OF LEARNNING OF CNN ARCHITECTURE
from tensorflow.keras.datasets import cifar10
from matplotlib import pyplot
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# create a grid of 3x3 images
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(X_train[i])
# Simple CNN model for the CIFAR-10 Dataset
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
#K.set_image_dim_ordering( 'th' )
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype( 'float32' )
X_test = X_test.astype( 'float32' )
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# Create the model
model = Sequential()
model.add(Convolution2D(32,(3, 3), input_shape=(32, 32, 3), activation= 'relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation= 'relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation= 'relu' , ))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation= 'softmax' ))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss= 'categorical_crossentropy' , optimizer=sgd, metrics=[ 'accuracy' ])
print(model.summary())
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,batch_size=32, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))



3) BUILD THE IMAGE CLASSIFICATION MODEL
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
y_train
class_names=['T-shirt/Top','Trosuser','pullover','dress','coat','sandle','shirt','sneakar','bag','ankle_boot']
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.imshow(x_train[i],cmap='gray')
  plt.title(class_names[y_train[i]])
  plt.xticks([])
x_train.shape
y_test.shape
x_test.shape
#features scale
x_train=x_train/255
x_test=x_test/255
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
model=Sequential([Flatten(input_shape=(28,28)),Dense(128,activation='relu'),Dense(10,activation='softmax')])
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=5)
test_loss,test_acc=model.evaluate(x_test,y_test)
new=x_train[345]
plt.imshow(new,cmap='gray')
predictions=model.predict(x_train)
data=np.argmax(predictions[345])
class_names[data]


4)ANOMALY DETECTION USING AUTOENCODER
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
path='''http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv'''
data=pd.read_csv(path,header=None)
print(data.shape)
data.head()
data.tail()
#last column is the target# 0= anomaly ,1 =normal
TARGET = 140
features=data.drop(TARGET,axis=1)
target=data[TARGET]
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=0,stratify=target)
x_train.shape
y_test.shape
x_test.shape
target.value_counts()
#use case is novelty detection so use only the normal for training
train_index=y_train[y_train==1].index
train_data=x_train.loc[train_index]
min_max_scaler=MinMaxScaler()
x_train_scaled=min_max_scaler.fit_transform(train_data.copy())
x_test_scaled=min_max_scaler.transform(x_test.copy())
x_train.describe()
pd.DataFrame(x_train_scaled).describe()
Build an AutoEncoder model
#create a model by subclassing Model class in tensorflow
class AutoEncoder(Model):
  """
  Parameters
  _____________
  output_units:int
  Number of output units
  code_size:int
  Number of units in bottle neck
  """
  def __init__(self,output_units,code_size=8):
    super().__init__()
    

    self.encoder=Sequential ([Dense(64,activation='relu'),Dropout(0.1),Dense(32,activation='relu'),Dropout(0.1),Dense(16,activation='relu'),Dropout(0.1),Dense(code_size,activation='relu')])
    self.decoder=Sequential ([Dense(16,activation='relu'),Dropout(0.1),Dense(32,activation='relu'),Dropout(0.1),Dense(64,activation='relu'),Dropout(0.1),Dense(output_units,activation='sigmoid')])
  def call(self,inputs):
   encoded=self.encoder(inputs)
   decoded=self.decoder(encoded)
   return decoded
model=AutoEncoder(output_units=x_train_scaled.shape[1])
#configurations of model
model.compile(loss='msle',metrics=['mse'],optimizer='adam')
history=model.fit(x_train_scaled,x_train_scaled,epochs=20,batch_size=512,validation_data=(x_test_scaled,x_test_scaled))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoches')
plt.ylabel('MSLE loss')
plt.legend(['loss','val_loss'])
plt.show()


5)IMPLEMENT THE CONTINOUS BAG OF WORDS
import re

import numpy as np 
import string

import pandas as pd 
import matplotlib as mpl

import matplotlib.pyplot as plt

from subprocess import check_output 
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
data ="""We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells."""

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(data)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 24))
axes[0].imshow(wordcloud)
axes[0].axis('off')
axes[1].imshow(wordcloud)
axes[1].axis('off')
axes[2].imshow(wordcloud)
axes[2].axis('off')
fig.tight_layout()

# Clean Data
sentences = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells."""

#strip it remove the space from the words
# remove special characters
sentences = re.sub('[^A-Za-z0-9]+', ' ', sentences)

# remove 1 letter words
sentences = re.sub(r'(?:^| )\w(?:$| )', ' ', sentences).strip()

# lower all characters
sentences = sentences.lower()

#Vocabulary
words = sentences.split()
vocab = set(words)

vocab_size = len(vocab)
embed_dim = 10
context_size = 2

# Creating the Dictionaries

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

ix_to_word
# DataBags

# data - [(context), target]

data = []
for i in range(2, len(words) - 2):
    context = [words[i - 2], words[i - 1], words[i + 1], words[i + 2]]
    target = words[i]
    data.append((context, target))
print(data[:5])
# Embeddings

embeddings =  np.random.random_sample((
    vocab_size, embed_dim))
# Linear Model
def linear(m, theta):
    w = theta
    return m.dot(w)
def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())
def NLLLoss(logs, targets):
    out = logs[range(len(targets)), targets]
    return -out.sum()/len(out)
def log_softmax_crossentropy_with_logits(logits,target):

    out = np.zeros_like(logits)
    out[np.arange(len(logits)),target] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    return (- out + softmax) / logits.shape[0]
def forward(context_idxs, theta):
    m = embeddings[context_idxs].reshape(1, -1)
    n = linear(m, theta)
    o = log_softmax(n)
    
    return m, n, o
def backward(preds, theta, target_idxs): 
    m, n, o = preds
    dlog = log_softmax_crossentropy_with_logits(n, target_idxs)
    dw = m.T.dot(dlog)
    return dw
def optimize(theta, grad, lr=0.03):
    theta -= grad * lr
    return theta
#Training

theta = np.random.uniform(-1, 1, (2 * context_size * embed_dim, vocab_size))
epoch_losses = {}

for epoch in range(80):

    losses =  []

    for context, target in data:
        context_idxs = np.array([word_to_ix[w] for w in context])
        preds = forward(context_idxs, theta)

        target_idxs = np.array([word_to_ix[target]])
        loss = NLLLoss(preds[-1], target_idxs)

        losses.append(loss)

        grad = backward(preds, theta, target_idxs)
        theta = optimize(theta, grad, lr=0.03)
        
     
    epoch_losses[epoch] = losses
ix = np.arange(0,80)
fig = plt.figure()
fig.suptitle('Epoch/Losses', fontsize=20)
plt.plot(ix,[epoch_losses[i][0] for i in ix])
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Losses', fontsize=12)

# predict funtion

def predict(words):
    context_idxs = np.array([word_to_ix[w] for w in words])
    preds = forward(context_idxs, theta)
    word = ix_to_word[np.argmax(preds[-1])]
    
    return word

# (['we', 'are', 'to', 'study'], 'about')
predict(['we', 'are', 'to', 'study'])

# more than 90% accuracy is not shown in nlp
# Accuracy

def accuracy():
    wrong = 0

    for context, target in data:
        if(predict(context) != target):
            wrong += 1
            
    return (1 - (wrong / len(data)))

accuracy()
