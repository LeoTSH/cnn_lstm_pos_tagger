
# coding: utf-8

# In[1]:


import keras, random, numpy as np, matplotlib.pyplot as plt
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


# In[2]:


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


# In[3]:


tb = TensorBoard(log_dir='./tf_logs/', batch_size=64, write_graph=True, histogram_freq=0)
data = open('./data/processed/ted_data', 'r', encoding='utf-8').read()
len(data)


# In[4]:


data = data.lower()
data_split = data.split('\n')
all_data = ' '.join(data_split)
words = all_data.split()


# In[5]:


# Chunk sequence
x = chunk_seq(words, 15)
sequences = [' '.join(seq) for seq in x]


# In[6]:


# Get sequence labels
process_labels = [get_labels(seq) for seq in sequences]
process_labels = [' '.join(seq) for seq in process_labels]


# In[7]:


# Check number of sequences and labels
print('Number of sequences: \t{}'.format(len(sequences)))
print('Number of labels: \t{}'.format(len(process_labels)))


# In[8]:


y_labels = open('./pos_labels', 'r').read()
y_labels = y_labels.split('\n')
all_labels = ' '.join(y_labels)
labels_tag = all_labels.split()


# In[9]:


# Build words vocab
words_in_vocab = Counter(words)
vocab = sorted(words_in_vocab, key=words_in_vocab.get, reverse=True)

# Skip most common word
vocab_to_int = {word: index for index, word in enumerate(vocab, 2)}
vocab_to_int['-PAD-'] = 0  # The special value used for padding
vocab_to_int['-OOV-'] = 1  # The special value used for OOVs
unique_vocab = len(vocab_to_int)
print('Number of unique words:', len(vocab_to_int))


# In[10]:


# Build labels vocab
labels_in_vocab = Counter(labels_tag)
labels_vocab = sorted(labels_in_vocab, key=labels_in_vocab.get, reverse=True)
label_to_int = {t: i for i, t in enumerate(labels_vocab, 1)}
label_to_int['-PAD-'] = 0  # The special value used to padding

# Check labels
no_classes = len(label_to_int)
print('Number of unique labels:', no_classes)
print(label_to_int)


# In[11]:


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


# In[12]:


# Check max seq length
seq_len = Counter([len(seq) for seq in seq_int])
max_seq_len = max(seq_len)
print("Maximum sequence length: {}".format(max(seq_len)))

encoded_labels = [to_categorical(i, num_classes=no_classes) for i in lab_int]
print(encoded_labels[10])


# In[13]:


# Pad sequences to 15 or sequence length, post padding
features = np.zeros((len(seq_int), 15), dtype=int)

for i, row in enumerate(seq_int):
    features[i, :len(row)] = np.array(row)[:15]

# Check that all sequences at at length 5
assert len(features)==len(seq_int)
assert len(features[0])==15


# In[14]:


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
print('Training Dataset: \t{}'.format(train_x.shape))
print('Validation Dataset: \t{}'.format(val_x.shape))
print('Testing Dataset: \t{}'.format(test_x.shape))


# In[15]:


model = Sequential()
# model.add(InputLayer(input_shape=(max_seq_len,)))
model.add(Embedding(input_dim=unique_vocab, output_dim=128, input_length=max_seq_len))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(no_classes, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy', ignore_class_accuracy(0)])
model.summary()
model.fit(x=train_x, y=np.array(train_y), batch_size=64, epochs=2, validation_data=(val_x, np.array(val_y)),
          shuffle=True, callbacks=[tb])


# In[ ]:


# scores = model.evaluate(x=test_x, y=np.array(test_y), verbose=1)
# print('Accuracy: {}'.format(scores[1] * 100))


# In[ ]:


pred = model.predict(test_x)
print(test_x[0])
print(pred[0])


# In[ ]:


# Sequence to predict
pred_x_seq = []
for x in test_x[400]:
    for i, v in vocab_to_int.items():
        if v == x:
            pred_x_seq.append(i)
print(' '.join(pred_x_seq))

# Predicted output
pred_y = pred[400].argmax(axis=1)
pred_y_seq = []
for x in pred_y:
    for i, v in label_to_int.items():
        if v == x:
            pred_y_seq.append(i)
print(' '.join(pred_y_seq))

