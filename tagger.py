import keras, string, random, datetime, numpy as np, matplotlib.pyplot as plt
from string import punctuation
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.utils import to_categorical
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

def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy

def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)]) 
        token_sequences.append(token_sequence) 
    return token_sequences

current = datetime.datetime.now()
date = current.strftime('%dd-%%mm')
tb = TensorBoard(log_dir='./tf_logs/{}'.format(current), batch_size=64, write_graph=True, histogram_freq=0)
table = str.maketrans('', '', punctuation)
max_seq_len = 20

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

sequences = [seq.translate(table) for seq in sequences]

with open('processed_input', 'w') as f:
    for x in sequences:
        f.write(x+'\n')

with open('processed_labels', 'w') as f:
    for x in process_labels:
        f.write(x+'\n')

# Check number of sequences and labels
print('Number of sequences: \t{}'.format(len(sequences)))
print('Number of labels: \t{}'.format(len(process_labels)))

y_labels = open('./processed_labels', 'r').read()
y_labels = y_labels.split('\n')
all_labels = ' '.join(y_labels)
labels_tag = all_labels.split()

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
print('Sample sequence:', sequences[10])
print('Sample sequence:', seq_int[10])

# Tokenize output labels
lab_int = []
for lab in y_labels:
    lab_int.append([label_to_int[word] for word in lab.split()])
print('Sample label:', lab_int[10])

# Check max seq length
print("Maximum sequence length: {}".format(max_seq_len))

# Pad sequences to max sequence length, post padding
features = np.zeros((len(seq_int), max_seq_len), dtype=int)

for i, row in enumerate(seq_int):
    features[i, :len(row)] = np.array(row)[:max_seq_len]

# Check that all sequences at at max sequence length 
assert len(features)==len(seq_int)
assert len(features[0])==max_seq_len

encoded_labels = [to_categorical(i, num_classes=no_classes) for i in lab_int]
# print(encoded_labels[10])

train_test_split_frac = 0.8
split_index = int(0.8*len(features))

# Split data into training, validation, and test data (features and labels, x and y)
train_x, left_over_x = features[:split_index], features[split_index:]
train_y, left_over_y = encoded_labels[:split_index], encoded_labels[split_index:]

val_test_index = int(0.5*len(left_over_x))
print('Validation/Test amount: \t{}'.format(val_test_index))

val_x, test_x = left_over_x[:val_test_index], left_over_x[val_test_index:]
val_y, test_y = left_over_y[:val_test_index], left_over_y[val_test_index:]

# print out the shapes of your resultant feature data
print('Training Dataset: \t{}'.format(train_x.shape))
print('Validation Dataset: \t{}'.format(val_x.shape))
print('Testing Dataset: \t{}'.format(test_x.shape))

model = Sequential()
model.add(Embedding(input_dim=unique_vocab, output_dim=128, input_length=max_seq_len))
model.add(Conv1D(filters=64, kernel_size=3, padding='SAME'))
model.add(Conv1D(filters=128, kernel_size=3, padding="SAME"))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(no_classes, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])#, ignore_class_accuracy(0)])
model.summary()
model.fit(x=train_x, y=np.array(train_y), batch_size=64, epochs=1, validation_data=(val_x, np.array(val_y)),
          shuffle=True, verbose=1, callbacks=[tb])

# print('Saving Model')
# model.save('model.h5')
# print('Done')

# scores = model.evaluate(x=test_x, y=np.array(test_y), verbose=1)
# print('Accuracy: {}'.format(scores[1] * 100))

# Sequence to predict
test_data = test_x[498]
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

for_report = model.predict(test_x)
test_y = np.array(test_y)

l = []
for x in test_y:
    b = []
    for s in x:
        b.append(s.argmax(axis=1))
    l.append(b)

print(for_report.argmax(axis=0))
print(l)
# cr = classification_report(y_true=np.argmax(test_y, axis=1), y_pred=np.argmax(for_report, axis=1))
# cm = confusion_matrix(y_true=np.array(test_y).argmax(axis=1), y_pred=np.argmax(for_report, axis=1))
# print('Classification Report:')
# print(cr)
# print('Confusion Matrix:')
# print(cm)