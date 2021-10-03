from keras.layers import Input, LSTM, Dense, concatenate
from keras.models import Model
from pickle import load
from keras.utils import plot_model
import matplotlib.pyplot as plt

#input format
# inp1 = [[[0.1], [0.2]], [[0.3], [0.4]]]
# inp2 = [[[0.1], [0.2]], [[0.5], [0.6]]]
# op = [1, 0]

maxlen = 15
glove_vec_size = 50

with open("trainable.pkl", "rb") as fp:
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = load(fp) 

x1_train = x1_train + x1_test
x2_train = x2_train + x2_test
y_train = y_train + y_test

q1 = Input(shape=(maxlen, glove_vec_size,), dtype='float32', name='question1')
lstm_q1 = LSTM(128)(q1)

q2 = Input(shape=(maxlen, glove_vec_size,), dtype='float32', name='question2')
lstm_q2 = LSTM(128)(q2)

sim_input = concatenate([lstm_q1, lstm_q2])
x = Dense(64, activation='relu')(sim_input)
sim_output = Dense(1, activation='sigmoid', name='sim_output')(x)
model = Model(inputs=[q1, q2], outputs=[sim_output])

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

plot_model(model, to_file='model.png', show_shapes=True)

history = model.fit([x1_train, x2_train], [y_train], epochs=10, validation_split=0.15)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()