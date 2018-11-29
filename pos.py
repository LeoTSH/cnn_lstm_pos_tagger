import keras, itertools, random, torch, numpy as np, matplotlib.pyplot as plt
from random import randint
from string import punctuation
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, InputLayer, Bidirectional, TimeDistributed, Activation

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

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

data = open('./data/processed/ted_data', 'r', encoding='utf-8').read()
len(data)

data = data.lower()
data_split = data.split('\n')
all_data = ' '.join(data_split)

words = all_data.split()

# Chunk sequence
x = chunk_seq(words, 15)
sequences = [' '.join(seq) for seq in x]

# Get sequence labels
process_labels = [get_labels(seq) for seq in sequences]
process_labels = [' '.join(seq) for seq in process_labels]

# with open('pos_labels', 'w', encoding='utf-8') as f:
#     for label in process_labels:
#         f.write(str(label)+'\n')

# Check number of sequences and labels
print('Number of sequences: \t{}'.format(len(sequences)))
print('Number of labels: \t{}'.format(len(process_labels)))

y_labels = open('./pos_labels', 'r').read()
y_labels = y_labels.split('\n')
all_labels = ' '.join(y_labels)
labels_tag = all_labels.split()

# Build words vocab
words_in_vocab = Counter(words)
vocab = sorted(words_in_vocab, key=words_in_vocab.get, reverse=True)

# Skip most common word
vocab_to_int = {word: index for index, word in enumerate(vocab, 2)}
vocab_to_int['-PAD-'] = 0  # The special value used for padding
vocab_to_int['-OOV-'] = 1  # The special value used for OOVs
unique_vocab = len(vocab_to_int)
print('Number of unique words:', len(vocab_to_int))

# Build labels vocab
labels_in_vocab = Counter(labels_tag)
labels_vocab = sorted(labels_in_vocab, key=labels_in_vocab.get, reverse=True)
label_to_int = {t: i for i, t in enumerate(labels_vocab, 1)}
label_to_int['-PAD-'] = 0  # The special value used to padding

# Check labels
no_classes = len(label_to_int)
print('Number of unique labels:', no_classes)
# print(label_to_int)

# for x in sequences[:10]:
#     print(x)

# for x in y_labels[:10]:
#     print(x)

# Tokenize input sequences
seq_int = []
for seq in sequences:
    seq_int.append([vocab_to_int[word] for word in seq.split()])
# print(seq_int[:10])

# Check max seq length
seq_len = Counter([len(seq) for seq in seq_int])
max_seq_len = max(seq_len)
print("Maximum sequence length: {}".format(max(seq_len)))

# Tokenize output labels
lab_int = []
for lab in y_labels:
    lab_int.append([label_to_int[word] for word in lab.split()])
# print(lab_int[:10])

# encoded_labels = np.array(lab_int)
# print('Encoded labels:', encoded_labels[:10])
# encoded_labels = to_categorical(y=encoded_labels, num_classes=4)
# print(encoded_labels[:10])

encoded_labels = to_categorical(lab_int, no_classes)
# print(encoded_labels[-1])
# Pad sequences to 5 or sequence length, post padding
features = np.zeros((len(seq_int), 15), dtype=int)

for i, row in enumerate(seq_int):
    features[i, :len(row)] = np.array(row)[:15]

# Check that all sequences at at length 5
assert len(features)==len(seq_int)
assert len(features[0])==15

# print('#############################')
# print('Features:', features[-1])
# print('#############################')
# print('Labels:', lab_int[-1])
# print('#############################')

train_test_split_frac = 0.8
split_index = int(0.8*len(features))

# Split data into training, validation, and test data (features and labels, x and y)
train_x, left_over_x = features[:split_index], features[split_index:]
train_y, left_over_y = encoded_labels[:split_index], encoded_labels[split_index:]

val_test_index = int(0.5*len(left_over_x))
print('Validation/Test amount: \t{}'.format(val_test_index))

val_x, test_x = left_over_x[:val_test_index], left_over_x[val_test_index:]
val_y, test_y = left_over_y[:val_test_index], left_over_y[val_test_index:]

## print out the shapes of your resultant feature data
print('Training Dataset: \t{}'.format(train_x.shape), train_y.shape)
print('Validation Dataset: \t{}'.format(val_x.shape), val_y.shape)
print('Testing Dataset: \t{}'.format(test_x.shape), test_y.shape)

model = Sequential()
model.add(Embedding(input_dim=unique_vocab, output_dim=128, input_length=max_seq_len))
# model.add(LSTM(units=256, return_sequences=True))
# model.add(Dense(no_classes, activation='softmax'))
# model.add(Bidirectional(LSTM(256, return_sequences=True)))
# model.add(TimeDistributed(Dense(no_classes, activation='softmax')))
# model.add(Activation('softmax')) 
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.summary()

model.fit(x=train_x, y=train_y, batch_size=64, epochs=10, validation_split=(val_x, val_y))
scores = model.evaluate(test_x, test_y)
print('Accuracy: {}'.format(scores[1] * 100))