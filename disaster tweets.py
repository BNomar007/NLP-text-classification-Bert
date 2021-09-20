# downloading a tokenization script created by the Google team
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

# import dependencies
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_colwidth', None)

import warnings
warnings.filterwarnings('ignore')

import nltk, re, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import tokenization


# Load the data
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')
test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

# train and text shapes
print(train_df.shape, test_df.shape)

# Check the duplicated tweets
dup_train = train_df['text'].duplicated().sum()
print(f'there are {dup_train} tweets duplicated in train_df.')

'''
it seems that we have 110 duplicated tweets based on text column
'''
# drop duplictes
train_df = train_df.drop_duplicates(subset=['text'], keep='first')

# new shape for train data
print(train_df.shape, test_df.shape)

# display first 5 rows
print(train_df.head(5))


# check the distribution of the disaster and no-disaster tweets
count = train_df['target'].value_counts()
sns.barplot(count.index, count)
count

# First 15 disaster tweets
for x in range(15):
    ex = train_df[train_df['target'] == 0]['text'][0:15].tolist()
    print(ex[x])

# First 15 non-disaster tweets
for x in range(15):
    ex = train_df[train_df['target'] == 1]['text'][0:15].tolist()
    print(ex[x])


# Cleaning the data by removing all special characters and  stopwords
def Data_Cleaning(text):
    text = text.lower()
    text = re.sub("won\'t", "will not", text)
    text = re.sub("can\'t", "can not", text)
    text = re.sub("don\'t", "do not", text)
    
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+',' ', text)
    text = re.sub(r'&amp?;',' ', text)
    text = re.sub(r'&lt;',' ', text)
    text = re.sub(r'&gt;',' ', text)
    
    text = re.sub(r'\d{2}:\d{2}:\d{2}', ' ', text)
    text = re.sub(r'UTC', ' ', text)
    text = re.sub(r'\d{2}km', ' ', text)
    text = re.sub(r"\b\d+\b", " ", text) # removing the numbers

    text = re.sub(r"#","",text) 
    text = re.sub(r"(?:\@)\w+", ' ', text)
    text = re.sub(r'\n', ' ', text)
    
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    text = re.sub(' +', ' ', text) # remove multiple spaces
    
    text = [word for word in word_tokenize(text) if not word in stopwords.words('english')]
    text = ' '.join(text)

    return text


# apply the cleaning function to the dataset and creating a new column of the cleaned data
train_df['cleaned'] = train_df['text'].apply(lambda x: Data_Cleaning(x))
test_df['cleaned'] = test_df['text'].apply(lambda x: Data_Cleaning(x))


# display the dataframe after cleaning the data
print(train_df.head())


	
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    masks = []
    segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        masks.append(pad_masks)
        segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(masks), np.array(segments)


# tokenizer from tokenization script
F_tokenizer = tokenization.FullTokenizer


# load the bert model from tfhub.dev
bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1', trainable=True)

to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
vocabulary = bert_layer.resolved_object.vocab_file.asset_path.numpy()

tokenizer = F_tokenizer(vocabulary, to_lower_case)


# creating a fucntion to build the model
def build_model(bert_layer, max_len=512):
    
    input_word_ids = Input(shape = (max_len,), dtype = tf.int32, name = "input_word_ids")
    input_mask = Input(shape = (max_len,), dtype = tf.int32, name = "input_mask")
    segment_ids = Input(shape = (max_len,), dtype = tf.int32, name = "segment_ids")

    pooled_sequence, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    output = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs = output)
    model.compile(Adam(lr=1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model


# extract the max length of the sentences
df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
max_length = len(max(df.cleaned, key=len))
print(max_length)

'''
It seems like the maximum lentgh of the cleaned tweets is 138 therefore we are going to use max len of 140
'''

train_input = bert_encode(train_df.cleaned.values, tokenizer, max_len=140)
test_input = bert_encode(test_df.cleaned.values, tokenizer, max_len=140)
train_labels = train_df['target'].values

model = build_model(bert_layer, max_len=140)
model.summary()

# creating a checkpoint to save the best val during the traininig
checkpoint = ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True)
train_history = model.fit(train_input, train_labels, validation_split = 0.25, epochs = 5, callbacks = [checkpoint], batch_size = 16)

model.load_weights('model.h5')
test_pred = model.predict(test_input)

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')


submission['target'] = test_pred.round().astype(int)
submission.to_csv('submission.csv', index=False)