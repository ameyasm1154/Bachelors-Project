import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from disease_dict import symptoms_dict

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
model.load_state_dict(torch.load('BERT_intent_classification.pt', map_location=lambda storage, loc: storage))

MAX_LEN = 16

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

intent_df = pd.read_csv('User_Intent_Patterns.csv')

def get_semantic_frame(user_input, global_context):

	intent = get_intent(user_input)

	print(intent)

	entities = get_entities(user_input)

	if len(entities['filenames']) > 0:
		intent = 'open_file_patterns'

	if len(entities['symptoms']) > 0:
		intent = 'inform_instruction'

	# TODO - multiple intents
	global_context['current_user_intents'].append(intent)
	global_context['current_user_entities'].append(entities)

	semantic_frame = {}
	semantic_frame['intent'] = ''
	semantic_frame['request_slots'] = {'answer':''}
	semantic_frame['inform_slots'] = {'instruction':'', 'task_name':'', 'context_name':''}

	semantic_frame['intent'] = 'inform'
	if intent not in ['open_file_patterns', 'inform_instruction']:
		semantic_frame['inform_slots']['task_name'] = intent
		if intent in ['diagnosis_patterns']:
			semantic_frame['inform_slots']['context_name'] = 'Not Needed'
		if intent in ['text_summarization_patterns', 'anomaly_detection_patterns']:
			semantic_frame['inform_slots']['instruction'] = 'Not Needed'
	elif intent == 'inform_instruction':
		semantic_frame['inform_slots']['instruction'] = user_input
	elif intent == 'open_file_patterns':
		semantic_frame['inform_slots']['context_name'] = entities['filenames'][-1]

	return semantic_frame, global_context

def get_intent(user_input):

	test_input = "[CLS] " + user_input + " [SEP]" 
	tokenized_text = tokenizer.tokenize(test_input)
	input_id = tokenizer.convert_tokens_to_ids(tokenized_text)
	input_id = pad_sequences([input_id], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

	selected_intent = -1
	max_similarity_score = 0.0
	for x in range(len(list(intent_df))):
		for known_value in intent_df[list(intent_df)[x]]:
			if type(known_value) == float:
				continue
			known_tokenized_text = tokenizer.tokenize("[CLS] " + known_value + " [SEP]")
			known_input_id = tokenizer.convert_tokens_to_ids(known_tokenized_text)
			known_input_id = pad_sequences([known_input_id], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

			current_similarity_score = cosine_similarity(input_id, known_input_id)[0][0]
			if current_similarity_score > max_similarity_score:
				max_similarity_score = current_similarity_score
				selected_intent = x

	
	intents = list(intent_df)

	if selected_intent != -1:
		print(max_similarity_score, intents[selected_intent])

	if selected_intent != -1 and max_similarity_score > 0.9999:
		return intents[selected_intent]

	input_id = torch.tensor(input_id)
	prediction = (model(input_id).detach().numpy())[0]
	print(prediction)

	if max(prediction) < 5.5:
		return 'inform_instruction'

	return intents[np.argmax(prediction)]

def get_entities(user_input):

	entities = {}
	entities['filenames'] = []
	entities['symptoms'] = []

	tokenized_text = tokenizer.tokenize(user_input)
	for token in user_input.split(' '):
		if token[-4:] in ['.txt', '.jpg', '.png']:
			entities['filenames'].append(token)

	for symptom in symptoms_dict.keys():
		if user_input.find(symptom.replace('_', ' ')) != -1:
			entities['symptoms'].append(symptom) 

	return entities