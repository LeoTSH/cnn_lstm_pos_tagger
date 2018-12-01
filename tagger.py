import keras, string, random, datetime, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from string import punctuation
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, InputLayer, Bidirectional, TimeDistributed, Activation

def chunk_seq(seq, chunk_len):
    chunked_seq = []
    for i in range(0, len(seq), chunk_len):
        chunked_seq.append(seq[i:i+chunk_len])
    return chunked_seq

def get_labels(seq):
    labels_seq = []
    seq = seq.split()
    for i in range(len(seq)):
        if ',' in seq[i]:
            labels_seq.append('<comma>')
        elif '.' in seq[i]:
            labels_seq.append('<period>')
        else:
            labels_seq.append('<na>')
    return labels_seq

# Set model parameters
max_seq_len = 30
no_filters_1 = 32
no_filters_2 = 64
kernel_1 = 3
kernel_2 = 3
lstm_hidden = 256
embed_dim = 128
adam_lr = 0.001
batch_size = 64
epochs = 2
valid_split = 0.3

# Set misc parameters
current = datetime.datetime.now()
date = current.strftime('%b-%d')
tb = TensorBoard(log_dir='./tf_logs/{}'.format(date), batch_size=64, write_graph=True, histogram_freq=0)

# Look-up table to remove punctuations from data
table = str.maketrans('', '', punctuation)

# Load and process input/label data
data = open('./data/processed/ted_data', 'r', encoding='utf-8').read()
data = data.lower()
data_split = data.split('\n')
all_data = ' '.join(data_split)
words = all_data.split()

# Chunk sequence
x = chunk_seq(words, max_seq_len)
sequences = [' '.join(seq) for seq in x]

# Get sequence labels
process_labels = [get_labels(seq) for seq in sequences]
process_labels = [' '.join(seq) for seq in process_labels]

# Remove punctuations
sequences = [seq.translate(table) for seq in sequences]

with open('./processed_input', 'w', encoding='utf-8') as f:
    for x in sequences:
        f.write(x+'\n')

with open('./processed_labels', 'w', encoding='utf-8') as f:
    for x in process_labels:
        f.write(x+'\n')

# Check number of sequences and labels
print('Number of sequences: \t{}'.format(len(sequences)))
print('Number of labels: \t{}'.format(len(process_labels)))

y_labels = open('./processed_labels', 'r', encoding='utf-8').read()
y_labels = y_labels.split('\n')
y_labels = y_labels[:-1]
all_labels = ' '.join(y_labels)
labels_tag = all_labels.split()

split = int(0.8*len(all_labels))
test_y_counts = all_labels[split:]
test_y_counts_split = test_y_counts.split()
counts = Counter(test_y_counts_split)

# Build words vocab
all_data = ' '.join(sequences)
words = all_data.split()
words_in_vocab = Counter(words)
vocab = sorted(words_in_vocab, key=words_in_vocab.get, reverse=True)

# Skip most common word
vocab_to_int = {word: index for index, word in enumerate(vocab, 2)}
vocab_to_int['-PAD-'] = 0  # The special value used for padding
vocab_to_int['-OOV-'] = 1  # The special value used for OOVs
unique_vocab = len(vocab_to_int)
print('Number of unique words:', unique_vocab)

# Build labels vocab
labels_in_vocab = Counter(labels_tag)
labels_vocab = sorted(labels_in_vocab, key=labels_in_vocab.get, reverse=True)
label_to_int = {t: i for i, t in enumerate(labels_vocab, 1)}
label_to_int['-PAD-'] = 0  # The special value used to padding

# Check labels
no_classes = len(label_to_int)
print('Class distribution:', Counter(labels_in_vocab))
print('Number of unique labels:', no_classes)
print(label_to_int)

# Tokenize input sequences
seq_int = []
for seq in sequences:
    seq_int.append([vocab_to_int[word] for word in seq.split()])

# Pad input sequences
pad_seq = pad_sequences(sequences=seq_int, maxlen=max_seq_len, padding='post', value=0)

# Check sample sequence
print('Sample sequence:', sequences[-1])
print('Sample sequence:', pad_seq[-1])

# Tokenize output labels
lab_int = []
for lab in y_labels:
    lab_int.append([label_to_int[word] for word in lab.split()])

# Pad input labels
pad_labels = pad_sequences(sequences=lab_int, maxlen=max_seq_len, padding='post', value=0)
encoded_labels = [to_categorical(i, num_classes=no_classes) for i in pad_labels]

# Check sample label
print('Sample label:', pad_labels[-1])
print('Encoded label', encoded_labels[-1])

# Check max seq length
print("Maximum sequence length: {}".format(max_seq_len))

# Check that all sequences and labels are at max sequence length 
assert len(pad_seq)==len(seq_int)
assert len(pad_seq[0])==max_seq_len

assert len(pad_labels)==len(lab_int)
assert len(pad_labels[0])==max_seq_len
print('Sequence and labels length check passed!')

# Split train and label dataset
train_test_split_frac = 0.8
split_index = int(0.8*len(pad_seq))

# Split data into training, validation, and test data (features and labels, x and y)
train_val_x, test_x = pad_seq[:split_index], pad_seq[split_index:]
train_val_y, test_y = encoded_labels[:split_index], encoded_labels[split_index:]

# print out the shapes of your resultant feature data
print('Training/Validation Dataset: \t{}'.format(train_val_x.shape), len(train_val_y))
print('Testing Dataset: \t\t{}'.format(test_x.shape), len(test_y))

# Model code
model = Sequential()
model.add(Embedding(input_dim=unique_vocab, output_dim=embed_dim, input_length=max_seq_len))
model.add(Conv1D(filters=no_filters_1, kernel_size=kernel_1, padding='SAME'))
model.add(Conv1D(filters=no_filters_2, kernel_size=kernel_2, padding="SAME"))
model.add(Bidirectional(LSTM(lstm_hidden, return_sequences=True)))
model.add(TimeDistributed(Dense(no_classes, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=Adam(adam_lr), metrics=['accuracy'])#, ignore_class_accuracy(0)])
model.summary()
model.fit(x=train_val_x, y=np.array(train_val_y), batch_size=batch_size, epochs=epochs, validation_split=valid_split, steps_per_epoch=None, validation_steps=None,
          shuffle=True, verbose=1, callbacks=[tb])

# print('Saving Model')
# model.save('model.h5')
# print('Done')

# scores = model.evaluate(x=test_x, y=np.array(test_y), verbose=1)
# print('Accuracy: {}'.format(scores[1] * 100))

# Make prediction on a single sequence
# Sequence to predict
test_data = test_x[255]
pred_x_seq = []
for x in test_data:
    for value, index in vocab_to_int.items():
        if x == index:
            pred_x_seq.append(value)

# Predicted output
pred_expand = model.predict(np.expand_dims(test_data, axis=0))
pred_y = []
for y in pred_expand:
    pred_y.append(np.argmax(y, axis=1))
print('Predictions Index:')
print(pred_y)

pred_y_seq = []
for x in pred_y:
    for y in x:
        for value, index in label_to_int.items():
            if y == index:
                pred_y_seq.append(value)

print('Prediction sequence:')            
print(' '.join(pred_x_seq))
print('Prediction output:')
print(' '.join(pred_y_seq))

# # WIP for CM and CR
for_report = model.predict(test_x)
out_pred = [np.argmax(x, axis=1) for x in for_report]
out_pred = np.concatenate(out_pred, axis=0)

y_ = [np.argmax(x, axis=1) for x in test_y]
y_ = np.concatenate(y_, axis=0)

print('Test dataset distribution:', counts)
cm = confusion_matrix(y_true=y_, y_pred=out_pred)
print(cm)

cr = classification_report(y_true=y_, y_pred=out_pred)
print(cr)