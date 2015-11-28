from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Merge, Permute, Dropout, TimeDistributedMerge
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
from time import time


"""
Train a memory network on the bAbI dataset.

References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1503.08895

- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895

Reaches 93% accuracy on task 'single_supporting_fact_10k' after 70 epochs.
Time per epoch: 3s on CPU (core i7).
"""


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y).astype(np.float32))


path = get_file('babi-tasks-v1-2.tar.gz',
                origin='http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
tar = tarfile.open(path)

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_train dtype:', inputs_train.dtype)
print('inputs_test shape:', inputs_test.shape)
print('inputs_test dtype:', inputs_test.dtype)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_train dtype:', queries_train.dtype)
print('queries_test shape:', queries_test.shape)
print('queries_test dtype:', queries_test.dtype)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_train dtype:', answers_train.dtype)
print('answers_test shape:', answers_test.shape)
print('answers_test dtype:', answers_test.dtype)
print('-')
t0 = time()
print('Compiling...')

EMBEDDING_SIZE = 64

# embed the input sequence into a sequence of vectors
input_for_attention = Sequential()
input_for_attention.add(Embedding(input_dim=vocab_size,
                        output_dim=EMBEDDING_SIZE,
                        input_length=story_maxlen))
# output: (samples, story_maxlen, embedding_dim)

# embed the question into a vector using a simple sum
# aggregate
question = Sequential()
question.add(Embedding(input_dim=vocab_size,
                       output_dim=EMBEDDING_SIZE,
                       input_length=query_maxlen))
# output: (samples, query_maxlen, embedding_dim)
question.add(TimeDistributedMerge(mode='sum'))

print('question shape:', question.output_shape)
# output: (samples, embedding_dim)

# compute a 'attention' between input sequence elements (which are vectors)
# and the question vector
attention = Sequential()
attention.add(Merge([input_for_attention, question],
                    mode='dot', dot_axes=[(2,), (1,)]))
attention.add(Activation('softmax'))
print('attention shape:', attention.output_shape)
# output: (samples, story_maxlen)

# embed the input into a single vector with size = story_maxlen:
input_for_response = Sequential()
input_for_response.add(Embedding(input_dim=vocab_size,
                                 output_dim=EMBEDDING_SIZE,
                                 input_length=story_maxlen))
# output: (samples, story_maxlen, embedding_dim)
# sum the attention vectors with the input vector:
response = Sequential()
response.add(Merge([attention, input_for_response],
                   mode='dot', dot_axes=[(1,), (1,)]))

print('response shape:', response.output_shape)
# output: (samples, embedding_dim)

# concatenate the attention vector with the question vector,
# and do logistic or mlp regression on top
answer = Sequential()
answer.add(Merge([response, question], mode='sum'))

# we output a probability distribution over the vocabulary
answer.add(Dense(vocab_size, activation='softmax'))

answer.compile(optimizer='adam', loss='categorical_crossentropy')
# Note: you could use a Graph model to avoid repeat the input twice
print('Model construction took %0.3fs' % (time() - t0))

answer.fit([inputs_train, queries_train, inputs_train], answers_train,
           batch_size=32,
           nb_epoch=500,
           show_accuracy=True,
           validation_data=([inputs_test, queries_test, inputs_test], answers_test))
