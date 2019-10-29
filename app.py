from flask import Flask, request, jsonify
from utils import generate_headlines
import tensorflow as tf
import json
tf.enable_eager_execution()

import numpy as np

app = Flask(__name__)

char2idx = json.loads(open("./data/st_char2idx.txt", 'br').read().decode(encoding='utf-8'))
idx2char = np.array(json.loads(open("./data/st_idx2char.txt", 'br').read().decode(encoding='utf-8')))
model = tf.keras.models.load_model('./models/st_model_20191004.h5')

def handle_error(status_code, message):
    """ Helper function to return handle HTTP errors
    and return message to users
    """
    response = jsonify({
        'status':status_code,
        'message':message,
    })    
    response.status_code = status_code
    return response


@app.route('/headlines', methods=['GET', 'POST'])
def headlines_generator():

    if request.method == "GET": 
        start_string = "Singapore "
        n_headlines = 5

    if request.method == "POST":
        data = request.json
        if not data:
            return handle_error(400, "POST data not found")

        start_string = data.get('start_string', 'Singapore ')
        n_headlines = data.get('n_headlines', 5) # default 5 headlines

        # Validate inputs
        if not isinstance(start_string, str):
            return handle_error(400, "start_string parameter must be string")

        if not isinstance(n_headlines, int):
            return handle_error(400, "n_headlines parameter must be integer")

        if len(start_string) > 256:
            # Limit number of start_string characters 
            return handle_error(400, "start_string character limit exceeded. Max character limit: 256")

        # limit to max 10 headlines
        # coerce negative and 0 to at least 1 headline
        n_headlines = min(max(n_headlines, 1), 10) 

    generated = generate_headlines(
        model, 
        char2idx,
        idx2char,
        start_string=start_string, 
        n_headlines=n_headlines
    )

    return jsonify(generated)