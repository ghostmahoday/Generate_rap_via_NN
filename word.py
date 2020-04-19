from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import random
import re
from unidecode import unidecode
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from random import randint

data = pd.read_csv('./songs.csv')

def tokenize_lines(df):
    words = []
    
    for index, row in df['lyrics'].iteritems():
        row = str(row).lower()
        for line in row.split('|-|'):
            new_words = re.findall(r"\b[a-z']+\b", unidecode(line))
            words = words + new_words
        
    return words


all_lines = tokenize_lines(data)

sequence_length = 51
sequences = []

for i in range(sequence_length, len(all_lines)):
    seq = all_lines[i - sequence_length: i]
    sequences.append(seq)


for i in sequences:
    data = ' '.join(i)
    
'\n'.join(data)
file = open('sequences.txt', 'w')
file.write(data)
file.close()

vocab = set(all_lines)

word_to_index = {w: i for i, w in enumerate(vocab)}
index_to_word = {i: w for w, i in word_to_index.items()}
word_indices = [word_to_index[word] for word in vocab]
vocab_size = len(vocab)

print('vocabulary size: {}'.format(vocab_size))


tokenized_seq = np.zeros((len(sequences), sequence_length))

for r, line in enumerate(sequences):
    for c, word in enumerate(line):
        tokenized_seq[r, c] = word_to_index[word]

tokenized_seq[:, -1].shape


X, y = tokenized_seq[:, :-1], tokenized_seq[:, -1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = len(X[0])


print("X_shape", X.shape)
print("y_shape", y.shape)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=50)

plt.plot(history.history['loss'])

model.save('model.h5')
model = load_model('model.h5')


def texts_to_sequences(texts, word_to_index):
    indices = np.zeros((1, len(texts)), dtype=int)
    
    for i, text in enumerate(texts):
        indices[:, i] = word_to_index[text]
        
    return indices


def pad_sequences(seq, maxlen):
    start = seq.shape[1] - maxlen
    
    return seq[:, start: start + maxlen]


def generate_seq(model, word_to_index, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text

    for _ in range(n_words):
        encoded = texts_to_sequences(in_text.split()[1:], word_to_index)
        encoded = pad_sequences(encoded, maxlen=seq_length)
        
        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''
    
        for word, index in word_to_index.items():
            if index == yhat:
                out_word = word
                break
        
        in_text += ' ' + out_word
        result.append(out_word)
        
    return ' '.join(result)

seed_text = 'i am beginning to feel like a rap god rap god all my friends say bad not all i wanna do is break free break free love my baby till love she i could care less but what can i say as she has all the till and i am just a nigga with no code'

generated = generate_seq(model, word_to_index, seq_length, 50)
print(generated)
