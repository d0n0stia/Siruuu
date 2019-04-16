#import keras
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from tensorflow.keras.layers import Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import numpy as np
import sys
import os

from data_help import Data_loader

# 하이퍼 파라미터
dl = Data_loader()

x_train, y_train = dl.data_load2('./data/train_data.pkl', './data/multihot_list.pkl')
#x_train, y_train = dl.data_load()
x_val = x_train[8000:]
y_val = y_train[8000:]
x_train = x_train[:8000]
y_train = y_train[:8000]
print(x_train.shape, ",", y_train.shape)
print(type(x_train), type(y_train))
seq_len = dl.max_length
print(seq_len)
output_length = dl.output_length
print(output_length)
epochs = 1000
batch_size = 32

# embedding size
embedding_dim = 100
hidden_dim = 50

# Filter parameters
filter_sizes = [5, 6, 7]
num_filters = 30

vocabulary_size = len(dl.word_to_index_dict)
print(vocabulary_size)
#input
inputs = Input(shape = (seq_len,), dtype = 'int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=seq_len)(inputs)
reshape = Reshape(( seq_len, embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', name='conv_0')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', name='conv_1')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', name='conv_2')(reshape)

maxpool_0 = MaxPool2D(pool_size=(seq_len - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid', name='maxpool_0')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(seq_len - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid', name='maxpool_1')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(seq_len - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid', name='maxpool_2')(conv_2)

cnn = Concatenate(axis=1, name='cnn_concat')([maxpool_0, maxpool_1, maxpool_2])
flat = Flatten(name='Flat')(cnn)

dropout = Dropout(0.5)(flat)
result = Dense(units=output_length, activation='softmax', name ='SOFTMAX')(dropout)

model = Model(inputs=inputs, outputs=result)
checkpoint = ModelCheckpoint('./result_train/weights.{epoch:03d}-{val_risk_softmax_acc:.4f}-{val_type_softmax_acc:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlystop = EarlyStopping(monitor='val_loss', patience=30, mode = 'auto', min_delta = 0)

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam,loss = 'categorical_crossentropy' ,metrics=['accuracy'])
# plot_model(model, to_file='architecture_model.png', show_shapes=True, show_layer_names=True)

print("Traning Model...")
#tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,  callbacks=[checkpoint, earlystop], validation_data=(x_val, y_val))  # starts training
