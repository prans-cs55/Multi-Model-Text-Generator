import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from datasets import load_dataset

# Hyperparameters
SEQ_LENGTH = 40
STEP_SIZE = 3
BATCH_SIZE = 128
EPOCHS = 3

def prepare_data(texts, max_chars=200000):
    print(f"Preparing data for {len(texts)} documents...")
    text = "\n".join(texts).lower()[:max_chars]
    chars = sorted(set(text))
    char_to_index = {c: i for i, c in enumerate(chars)}
    index_to_char = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)

    sentences = []
    next_chars = []
    for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
        sentences.append(text[i:i+SEQ_LENGTH])
        next_chars.append(text[i+SEQ_LENGTH])

    print(f"Number of sequences: {len(sentences)}")

    x = np.zeros((len(sentences), SEQ_LENGTH, vocab_size), dtype=bool)
    y = np.zeros((len(sentences), vocab_size), dtype=bool)
    for i, sentence in enumerate(sentences):
        for t, ch in enumerate(sentence):
            x[i, t, char_to_index[ch]] = True
        y[i, char_to_index[next_chars[i]]] = True

    return x, y, char_to_index, index_to_char, vocab_size, text

def build_model(seq_length, vocab_size):
    model = Sequential([
        tf.keras.Input(shape=(seq_length, vocab_size)),
        LSTM(256),
        Dense(vocab_size),
        Activation('softmax'),
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=0.01))
    return model

def train_and_save_model(texts, model_name):
    x, y, cti, itc, vocab_size, text = prepare_data(texts)
    model = build_model(SEQ_LENGTH, vocab_size)
    model.summary()
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS)
    model.save(f"{model_name}.keras")
    return model, cti, itc, text

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, preds, 1))

def generate_text(model, char_to_index, index_to_char, text, length=300, temperature=0.5):
    start_idx = random.randint(0, len(text) - SEQ_LENGTH - 1)
    sentence = text[start_idx:start_idx + SEQ_LENGTH]
    generated = sentence
    for _ in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(char_to_index)))
        for t, ch in enumerate(sentence):
            x_pred[0, t, char_to_index[ch]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample(preds, temperature)
        next_char = index_to_char[next_idx]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

# Load Shakespeare text from TensorFlow URL
print("Loading Shakespeare dataset...")
filepath = tf.keras.utils.get_file(
    'shake.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)
with open(filepath, 'r', encoding='utf-8') as f:
    shakespeare_text = f.read()

# Treat Shakespeare text as a single document (list with one element)
poetry_texts = [shakespeare_text]

print("Loading lyrics dataset...")
lyrics_ds = load_dataset("brunokreiner/genius-lyrics", split="train[:500]")
lyrics_texts = [item['lyrics'] for item in lyrics_ds if item.get('lyrics') and item['lyrics'].strip() != '']
print(f"Loaded {len(lyrics_texts)} lyrics texts")

print("Loading blog dataset...")
blog_ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:500]")
blog_texts = [item['text'] for item in blog_ds if item.get('text') and item['text'].strip() != '']
print(f"Loaded {len(blog_texts)} blog texts")

print("Training poetry model...")
poetry_model, poetry_cti, poetry_itc, poetry_text = train_and_save_model(poetry_texts, "poetry_model")

print("Training lyrics model...")
lyrics_model, lyrics_cti, lyrics_itc, lyrics_text = train_and_save_model(lyrics_texts, "lyrics_model")

print("Training blog model...")
blog_model, blog_cti, blog_itc, blog_text = train_and_save_model(blog_texts, "blog_model")

print("\nGenerated poetry text:")
print(generate_text(poetry_model, poetry_cti, poetry_itc, poetry_text))

print("\nGenerated lyrics text:")
print(generate_text(lyrics_model, lyrics_cti, lyrics_itc, lyrics_text))

print("\nGenerated blog text:")
print(generate_text(blog_model, blog_cti, blog_itc, blog_text))
