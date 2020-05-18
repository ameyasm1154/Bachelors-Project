import math
import os
import numpy as np
import pandas as pd
import pickle
import skimage
import pickle as pkl
import cv2
import re
from collections import Counter
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
import tensorflow as tf

vgg_path = 'vgg16-20160129.tfmodel'
visual_qa_img_base_path = 'static/'
word_to_idx = pickle.load(open('word_to_idx.pkl', 'rb'))
idx_to_word = pickle.load(open('idx_to_word.pkl', 'rb'))
visual_qa_df = pickle.load(open('test_df_v_final.pkl', 'rb'))
visual_qa_features = pickle.load(open('image_feature_test', 'rb'))
vocabulary_size = len(word_to_idx)
BUFFER_TOKENS = ['<NULL>', '<START>', '<END>', '<UNK>']
PADDING_LEN = 17

def parse_sentence(s):
    s = s.replace('.', '')
    s = s.replace(',', '')
    s = s.replace('"', '')
    s = s.replace("'", '')
    s = s.replace("?", '')
    s = s.lower()
    s = re.sub("\s\s+", " ", s)
    s = s.split(' ')
    return s

def _convert_sentence_to_numbers(s):
    UNK_IDX = BUFFER_TOKENS.index('<UNK>')
    NULL_IDX = BUFFER_TOKENS.index('<NULL>')
    END_IDX = BUFFER_TOKENS.index('<END>')
    STR_IDX = BUFFER_TOKENS.index('<START>')
    s_encoded = [word_to_idx.get(w, UNK_IDX) for w in s]
    s_encoded = [STR_IDX] + s_encoded
    s_encoded += [END_IDX]
    s_encoded += [NULL_IDX] * (PADDING_LEN - 1 - len(s_encoded))
    return s_encoded

def decode_sequence(image_features, input_seq):
    # Encode the input as state vectors.
    encoder_model = load_model('encoder.h5')
    decoder_model = load_model('decoder.h5')
    states_value = encoder_model.predict([image_features,input_seq])

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, vocabulary_size))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 1] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = idx_to_word[sampled_token_index]
        decoded_sentence += sampled_char
        decoded_sentence +=" "

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '2' or
           len(decoded_sentence) > PADDING_LEN-1):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, vocabulary_size))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def find_end(item,st):
    count = 0
    for i in range(0,len(item)):
        count = i
        if item[i]==st:
            return i
    return count+1 

def get_answer_for_image(question, image_filename):
	print(image_filename[:-4])
	image_indices = visual_qa_df.index[visual_qa_df['Image'] == image_filename[:-4]].tolist()
	image_index = image_indices[0]
	image_input = visual_qa_features[image_index]
	image_input = np.array(visual_qa_features[0])
	image_input = image_input.reshape(1, image_input.shape[0])
	question_input = np.array(_convert_sentence_to_numbers(parse_sentence(question)))
	question_input = question_input.reshape(1, question_input.shape[0])
	decoded_sentence = decode_sequence(image_input, question_input).split(' ')
	answer = ''
	decoded_sentence = decoded_sentence[0:find_end(decoded_sentence,'<END>')]
	for itr in range(len(decoded_sentence)): 
	    answer += decoded_sentence[itr] 
	    if itr != len(decoded_sentence)-1:
	        answer += ' '

	return answer