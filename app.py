from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained lyrics generation model
model = tf.keras.models.load_model('hillsong_lyrics_model.h5')

# Load the tokenizer used to train the model
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_lyrics', methods=['POST'])
def generate_lyrics():
    # Get the seed text and length of lyrics to generate from the form submission
    seed_text = request.form['seed_text']
    length = int(request.form['length'])
    
    # Generate the lyrics based on the seed text and length
    generated_lyrics = generate_text(model, tokenizer, seed_text, length)
    
    # Render the generated lyrics in the HTML template
    return render_template('index.html', generated_lyrics=generated_lyrics)

def generate_text(model, tokenizer, seed_text, length):
    # Generate the next `length` words/characters of lyrics based on the seed text
    for i in range(length):
        # Convert the current lyrics sequence to a sequence of token IDs
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Pad the sequence to a fixed length to match the input shape of the model
        token_list = pad_sequences([token_list], maxlen=11, padding='pre')
        
        # Predict the next token ID using the model
        predicted_token = np.argmax(model.predict(token_list), axis=-1)[0]
        
        # Convert the predicted token ID back to a word/character and add it to the generated lyrics
        output_word = tokenizer.index_word[predicted_token]
        seed_text += " " + output_word
        
    return seed_text

if __name__ == '__main__':
    app.run(debug=True)
