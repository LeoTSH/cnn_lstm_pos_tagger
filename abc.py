import nltk
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam

tagged_sentences = nltk.corpus.treebank.tagged_sents()
print(tagged_sentences[0])
print("Tagged sentences: ", len(tagged_sentences))
print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))
# [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')]
# Tagged sentences:  3914
# Tagged words: 100676

sentences, sentence_tags =[], [] 
for tagged_sentence in tagged_sentences:
    sentence, tags = zip(*tagged_sentence)
    sentences.append(np.array(sentence))
    sentence_tags.append(np.array(tags))
# Let's see how a sequence looks

print(sentences[5])
print(sentence_tags[5])

# ['Lorillard' 'Inc.' ',' 'the' 'unit' 'of' 'New' 'York-based' 'Loews'
#  'Corp.' 'that' '*T*-2' 'makes' 'Kent' 'cigarettes' ',' 'stopped' 'using'
#  'crocidolite' 'in' 'its' 'Micronite' 'cigarette' 'filters' 'in' '1956'
# '.']
# ['NNP' 'NNP' ',' 'DT' 'NN' 'IN' 'JJ' 'JJ' 'NNP' 'NNP' 'WDT' '-NONE-' 'VBZ'
#  'NNP' 'NNS' ',' 'VBD' 'VBG' 'NN' 'IN' 'PRP$' 'NN' 'NN' 'NNS' 'IN' 'CD'
#  '.']

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

(train_sentences, 
test_sentences, 
train_tags, 
test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)

words, tags = set([]), set([])

for s in train_sentences:
    for w in s:
        words.add(w.lower())

for ts in train_tags:
    for t in ts:
        tags.add(t)

word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs
tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding

train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []
for s in train_sentences:
    s_int = []

for w in s:
    try:
       s_int.append(word2index[w.lower()])
    except KeyError:
        s_int.append(word2index['-OOV-'])
train_sentences_X.append(s_int)

for s in test_sentences:
    s_int = []

for w in s:
    try:
        s_int.append(word2index[w.lower()])
    except KeyError:
        s_int.append(word2index['-OOV-'])
test_sentences_X.append(s_int)

for s in train_tags:
    train_tags_y.append([tag2index[t] for t in s])

for s in test_tags:
    test_tags_y.append([tag2index[t] for t in s])

print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])
# [2385, 9167, 860, 4989, 6805, 6349, 9078, 3938, 862, 1092, 4799, 860, 1198, 1131, 879, 5014, 7870, 704, 4415, 8049, 9444, 8175, 8172, 10058, 10034, 9890, 1516, 8311, 7870, 1489, 7967, 6458, 8859, 9720, 6754, 5402, 9254, 2663]
# [3829, 3347, 1, 8311, 6240, 982, 7936, 1, 3552, 4558, 1, 9007, 8175, 8172, 637, 4517, 7392, 3124, 860, 5416, 920, 3301, 6240, 1205, 5282, 6683, 9890, 758, 4415, 1, 6240, 3386, 9072, 3219, 6240, 9157, 5611, 6240, 6969, 4517, 2956, 175, 2663]
# [11, 35, 39, 3, 7, 9, 20, 42, 42, 3, 35, 39, 35, 35, 22, 7, 10, 16, 32, 35, 31, 17, 3, 11, 42, 7, 9, 3, 10, 16, 6, 25, 12, 11, 42, 17, 6, 44]
# [2, 35, 16, 3, 20, 35, 42, 42, 16, 25, 7, 31, 17, 3, 35, 15, 42, 7, 39, 35, 35, 16, 20, 42, 40, 16, 7, 6, 32, 30, 20, 42, 42, 37, 20, 42, 3, 20, 42, 15, 11, 42, 44]

MAX_LENGTH = len(max(train_sentences_X, key=len))
print(MAX_LENGTH)  # 271

train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')
print(train_sentences_X[0])
print(test_sentences_X[0])
print('##################')
print(train_tags_y[0])
print(test_tags_y[0])

model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding_1 (Embedding)      (None, 271, 128)          1302400   
# _________________________________________________________________
# bidirectional_1 (Bidirection (None, 271, 512)          788480    
# _________________________________________________________________
# time_distributed_1 (TimeDist (None, 271, 47)           24111     
# _________________________________________________________________
# activation_1 (Activation)    (None, 271, 47)           0         
# =================================================================
# Total params: 2,114,991
# Trainable params: 2,114,991
# Non-trainable params: 0
# _________________________________________________________________

model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=10, validation_split=0.2)
scores = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
print('Accuracy: {}'.format(scores[1*100]))   # acc: 99.09751977804825