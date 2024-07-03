#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')
get_ipython().system('pip install numpy==1.17')
get_ipython().system('pip install pandas')
get_ipython().system('pip install nltk')
get_ipython().system('pip install rouge-score')
get_ipython().system('pip install tqdm')


# In[2]:


import os
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK punkt package
nltk.download('punkt')

# Load the dataset
data_dir = 'C:\\Users\\acer\\Downloads\\archive (1)\\cnn_dailymail'  # Path to your dataset directory
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
val_data = pd.read_csv(os.path.join(data_dir, 'validation.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Sample the data for testing (using only 1000 rows)
train_data = train_data.sample(n=1000, random_state=42)
val_data = val_data.sample(n=200, random_state=42)
test_data = test_data.sample(n=200, random_state=42)

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

train_data['article'] = train_data['article'].apply(preprocess_text)
train_data['highlights'] = train_data['highlights'].apply(preprocess_text)
val_data['article'] = val_data['article'].apply(preprocess_text)
val_data['highlights'] = val_data['highlights'].apply(preprocess_text)
test_data['article'] = test_data['article'].apply(preprocess_text)
test_data['highlights'] = test_data['highlights'].apply(preprocess_text)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.concat([train_data['article'], train_data['highlights']]))

vocab_size = len(tokenizer.word_index) + 1
max_article_len = 400
max_summary_len = 100

train_article_seq = pad_sequences(tokenizer.texts_to_sequences(train_data['article']), maxlen=max_article_len, padding='post')
train_summary_seq = pad_sequences(tokenizer.texts_to_sequences(train_data['highlights']), maxlen=max_summary_len, padding='post')

val_article_seq = pad_sequences(tokenizer.texts_to_sequences(val_data['article']), maxlen=max_article_len, padding='post')
val_summary_seq = pad_sequences(tokenizer.texts_to_sequences(val_data['highlights']), maxlen=max_summary_len, padding='post')

test_article_seq = pad_sequences(tokenizer.texts_to_sequences(test_data['article']), maxlen=max_article_len, padding='post')
test_summary_seq = pad_sequences(tokenizer.texts_to_sequences(test_data['highlights']), maxlen=max_summary_len, padding='post')

print("Data preprocessing completed.")


# In[3]:


import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

embedding_dim = 128
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(max_article_len,))
enc_emb = Embedding(vocab_size, embedding_dim, trainable=True)(encoder_inputs)

encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output, state_h, state_c = encoder_lstm(enc_emb)

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(vocab_size, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_output, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# Attention Mechanism
attention = tf.keras.layers.Attention()
attention_output = attention([decoder_output, encoder_output])

# Concatenate context vector and decoder output
decoder_combined_context = Concatenate(axis=-1)([attention_output, decoder_output])

# Final output layer
output_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))
decoder_output = output_layer(decoder_combined_context)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_output)
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy')
model.summary()

# Define data generator
def data_generator(articles, summaries, batch_size, max_summary_len):
    while True:
        for i in range(0, len(articles), batch_size):
            encoder_input_data = articles[i:i + batch_size]
            decoder_input_data = summaries[i:i + batch_size, :-1]
            decoder_target_data = summaries[i:i + batch_size, 1:]
            yield ([encoder_input_data, decoder_input_data], decoder_target_data)


# In[4]:


import pandas as pd

# Define batch size and epochs
batch_size = 32
epochs = 10

# Define the paths to your CSV files
train_csv_path = r"C:\Users\acer\Downloads\archive (1)\cnn_dailymail\train.csv"
val_csv_path = r"C:\Users\acer\Downloads\archive (1)\cnn_dailymail\validation.csv"
test_csv_path = r"C:\Users\acer\Downloads\archive (1)\cnn_dailymail\test.csv"

# Load data from CSV files
train_data = pd.read_csv(train_csv_path)
val_data = pd.read_csv(val_csv_path)
test_data = pd.read_csv(test_csv_path)

# Display the first few rows of each dataset to verify the data loading
print("Train data:")
print(train_data.head())
print("\nValidation data:")
print(val_data.head())
print("\nTest data:")
print(test_data.head())


# In[5]:


import os
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define tokenizer parameters
num_words = 5000  # Adjust as necessary
max_article_len = 400  # Adjust as necessary
max_summary_len = 50  # Adjust as necessary

# Define paths to save tokenized sequences
tokenized_data_dir = r"C:\Users\acer\Downloads\archive (1)\cnn_dailymail\tokenized"
os.makedirs(tokenized_data_dir, exist_ok=True)

article_tokenizer_path = os.path.join(tokenized_data_dir, 'article_tokenizer.json')
summary_tokenizer_path = os.path.join(tokenized_data_dir, 'summary_tokenizer.json')

# Initialize tokenizers
article_tokenizer = Tokenizer(num_words=num_words)
summary_tokenizer = Tokenizer(num_words=num_words)

# Fit tokenizers on training data
article_tokenizer.fit_on_texts(train_data['article'])
summary_tokenizer.fit_on_texts(train_data['highlights'])

# Save tokenizers
with open(article_tokenizer_path, 'w') as f:
    f.write(article_tokenizer.to_json())
with open(summary_tokenizer_path, 'w') as f:
    f.write(summary_tokenizer.to_json())

# Function to tokenize and pad sequences, and save them
def tokenize_and_save(data, tokenizer, max_len, file_path):
    sequences = tokenizer.texts_to_sequences(data)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    np.save(file_path, padded_sequences)

# Tokenize and save sequences
tokenize_and_save(train_data['article'], article_tokenizer, max_article_len, os.path.join(tokenized_data_dir, 'train_article_seq.npy'))
tokenize_and_save(train_data['highlights'], summary_tokenizer, max_summary_len, os.path.join(tokenized_data_dir, 'train_summary_seq.npy'))
tokenize_and_save(val_data['article'], article_tokenizer, max_article_len, os.path.join(tokenized_data_dir, 'val_article_seq.npy'))
tokenize_and_save(val_data['highlights'], summary_tokenizer, max_summary_len, os.path.join(tokenized_data_dir, 'val_summary_seq.npy'))
tokenize_and_save(test_data['article'], article_tokenizer, max_article_len, os.path.join(tokenized_data_dir, 'test_article_seq.npy'))
tokenize_and_save(test_data['highlights'], summary_tokenizer, max_summary_len, os.path.join(tokenized_data_dir, 'test_summary_seq.npy'))

print("Tokenization and saving completed.")


# In[6]:


import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load tokenizers for sample data
with open(os.path.join(tokenized_data_dir, 'article_tokenizer.json'), 'r') as f:
    article_tokenizer_config = json.load(f)
    article_tokenizer = tokenizer_from_json(json.dumps(article_tokenizer_config))

with open(os.path.join(tokenized_data_dir, 'summary_tokenizer.json'), 'r') as f:
    summary_tokenizer_config = json.load(f)
    summary_tokenizer = tokenizer_from_json(json.dumps(summary_tokenizer_config))

print("Tokenizers loaded for sample data.")


# In[7]:


import tensorflow as tf

# Define data generator
def data_generator(article_seq, summary_seq, batch_size, max_summary_len):
    while True:
        for i in range(0, len(article_seq), batch_size):
            encoder_input_data = article_seq[i:i + batch_size]
            decoder_input_data = summary_seq[i:i + batch_size, :-1]
            decoder_target_data = summary_seq[i:i + batch_size, 1:]

            yield (
                [tf.convert_to_tensor(encoder_input_data, dtype=tf.float32), 
                 tf.convert_to_tensor(decoder_input_data, dtype=tf.float32)], 
                tf.convert_to_tensor(decoder_target_data, dtype=tf.float32)
            )


# In[8]:


# Define output signature
output_signature = (
    (
        tf.TensorSpec(shape=(None, max_article_len), dtype=tf.float32),
        tf.TensorSpec(shape=(None, max_summary_len-1), dtype=tf.float32),
    ),
    tf.TensorSpec(shape=(None, max_summary_len-1), dtype=tf.float32)
)

# Create datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_article_seq, train_summary_seq, batch_size, max_summary_len),
    output_signature=output_signature
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_article_seq, val_summary_seq, batch_size, max_summary_len),
    output_signature=output_signature
)


# In[9]:


import os
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Model

# Download NLTK punkt package
nltk.download('punkt')

# Load the dataset
data_dir = 'C:\\Users\\acer\\Downloads\\archive (1)\\cnn_dailymail'  # Path to your dataset directory
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))

# Sample a small portion of the dataset for testing (100 rows)
train_sample = train_data.sample(n=100, random_state=42)

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

train_sample['article'] = train_sample['article'].apply(preprocess_text)
train_sample['highlights'] = train_sample['highlights'].apply(preprocess_text)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.concat([train_sample['article'], train_sample['highlights']]))

vocab_size = len(tokenizer.word_index) + 1
max_article_len = 400
max_summary_len = 100

train_article_seq = pad_sequences(tokenizer.texts_to_sequences(train_sample['article']), maxlen=max_article_len, padding='post')
train_summary_seq = pad_sequences(tokenizer.texts_to_sequences(train_sample['highlights']), maxlen=max_summary_len, padding='post')

train_summary_seq_input = train_summary_seq[:, :-1]
train_summary_seq_target = train_summary_seq[:, 1:]

# Define the model
embedding_dim = 128
latent_dim = 512

# Encoder
encoder_inputs = Input(shape=(max_article_len,))
enc_emb = Embedding(vocab_size, embedding_dim, trainable=True)(encoder_inputs)

encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(vocab_size, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare data for training
train_summary_seq_input = train_summary_seq_input.reshape((train_summary_seq_input.shape[0], train_summary_seq_input.shape[1]))
train_summary_seq_target = train_summary_seq_target.reshape((train_summary_seq_target.shape[0], train_summary_seq_target.shape[1], 1))

# Train the model
batch_size = 32
epochs = 5

model.fit(
    [train_article_seq, train_summary_seq_input], 
    train_summary_seq_target,  
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)

print("Model training completed.")


# In[10]:


# Load the "test" dataset
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Sample a small portion of the test dataset for evaluation (100 rows)
test_sample = test_data.sample(n=500, random_state=42)

# Preprocess text for the test dataset
test_sample['article'] = test_sample['article'].apply(preprocess_text)
test_sample['highlights'] = test_sample['highlights'].apply(preprocess_text)

# Tokenize and pad sequences for the test dataset using the same tokenizer used for training
test_article_seq = pad_sequences(tokenizer.texts_to_sequences(test_sample['article']), maxlen=max_article_len, padding='post')
test_summary_seq = pad_sequences(tokenizer.texts_to_sequences(test_sample['highlights']), maxlen=max_summary_len, padding='post')

# Prepare test data for evaluation
test_summary_seq_input = test_summary_seq[:, :-1]
test_summary_seq_target = test_summary_seq[:, 1:]

# Reshape test data for compatibility with the model
test_summary_seq_input = test_summary_seq_input.reshape((test_summary_seq_input.shape[0], test_summary_seq_input.shape[1]))
test_summary_seq_target = test_summary_seq_target.reshape((test_summary_seq_target.shape[0], test_summary_seq_target.shape[1], 1))

# Evaluate the model on the test data
test_loss = model.evaluate([test_article_seq, test_summary_seq_input], test_summary_seq_target)

print(f"Test Loss: {test_loss}")


# In[ ]:




