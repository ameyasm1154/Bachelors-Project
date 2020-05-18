import json

def get_natural_language_response(semantic_frame, global_context):

	agent_response = json.loads(semantic_frame.replace('\'', '\"'))

	natural_language_response = ''

	if agent_response['intent'] in ['request', 'greet']:

		added_count = 0
		natural_language_response = 'please provide me -'
		for key in agent_response['request_slots']:
			if agent_response['request_slots'][key] != '':
				if added_count > 1:
					natural_language_response += ', '+str(key).replace('_', ' ')
				else:
					natural_language_response += ' '+str(key).replace('_', ' ')
			added_count += 1

	elif agent_response['intent'] == 'inform':

		if global_context['constant_user_informs']['task_name'] == 'diagnosis_patterns':

			answer = json.loads(agent_response['inform_slots']['answer'].replace('\'', '\"'))

			natural_language_response = 'The diagnosis is '+answer['diagnosis']+' with '+str(float(answer['confidence'])*100)+'% confidence'

		else:

			natural_language_response = agent_response['inform_slots']['answer']

	return natural_language_response