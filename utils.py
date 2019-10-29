import tensorflow as tf
tf.enable_eager_execution()

def generate_headlines(model, char2idx, idx2char, start_string=u"Singapore ", n_headlines=5):
    '''
    Function to generate headline predictions given a starting text.
  
    Parameters
    ----------
    model : obj
        Tensorflow model object
    char2idx : dict
        Dictionary mapping of characters to id
    idx2char : np.array
        numpy array of characters to map id back to char
    start_string : str
        Starting string as input to RNN. Will be the start of first headline
    n_headlines : int
        Number of headlines to generate
    
    Returns
    -------
    list
        Returns list of string headlines. Number of items in list
        corresponds to n_headlines generated
    '''  
    
    headline_count = 0
    input_eval = [char2idx.get(s, char2idx[' ']) for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    # Start first headline with user given text
    text_generated = start_string   
    headlines = []
    temperature = 0.6
    model.reset_states()

    while headline_count < n_headlines:
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        char_generated = idx2char[predicted_id]
        if char_generated == '\n':
            headline_count += 1
            headlines.append(text_generated)
            text_generated = ''
        else:
            text_generated += char_generated
    return headlines

