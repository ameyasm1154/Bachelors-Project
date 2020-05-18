from dialogue_config import FAIL, SUCCESS, REWARD, PENALTY, NO_OUTCOME, user_sim_default_request, user_intents, all_slots
from utils import reward_function
import json

class User:

	def __init__(self, constants, global_context):

		self.max_round = constants['run']['max_round_num']
		self.constants = constants

	# def reset(self, global_context):

	# 	return self.get_response_and_take_action(global_context)

	def get_response_and_take_action(self, global_context, user_input):

		# TODO - add user input error handling later

		# user_input = input('User : ')

		# user_input = json.loads(user_input.replace('\'', '\"'))

		global_context['controller'] = 'user'

		if user_input['intent'] == 'inform':
			global_context['current_user_intents'] = ['inform']
			global_context['current_user_informs'] = user_input['inform_slots']
			global_context['all_user_informs'].append(user_input['inform_slots'])
			for key in global_context['constant_user_informs'].keys():
				if user_input['inform_slots'][key] != '':
					if global_context['constant_user_informs'][key] == '':
						global_context['constant_user_informs'][key] = user_input['inform_slots'][key]
					elif global_context['constant_user_informs'][key] != '':
						if global_context['constant_user_informs'][key] != user_input['inform_slots'][key]:
							global_context['constant_user_informs'][key] = user_input['inform_slots'][key]
							print('Agent: Replacing context of {}'.format(key))
							if key == 'task_name':
								global_context['constant_user_informs']['instruction'] = ''
								if global_context['constant_user_informs']['task_name'] in ['diagnosis_patterns']:
									global_context['constant_user_informs']['context_name'] = 'Not Needed'
								if global_context['constant_user_informs']['task_name'] in ['text_summarization_patterns', 'anomaly_detection_patterns']:
									global_context['constant_user_informs']['instruction'] = 'Not Needed'
							elif key == 'context_name':
								global_context['constant_user_informs']['instruction'] = ''
								global_context['constant_user_informs']['task_name'] = ''

		elif user_input['intent'] == 'request':
			global_context['current_user_intents'] = ['request']
			global_context['current_user_requests'] = user_input['request_slots']
			global_context['all_user_requests'].append(user_input['request_slots'])
			for key in global_context['constant_user_requests']:
				if user_input['request_slots'][key] != '':
					global_context['constant_user_requests'][key] = user_input['request_slots'][key]

		elif user_input['intent'] == 'done_success':
			global_context['current_user_intents'] = ['done_success']
			global_context['current_state'] = 'done'

		elif user_input['intent'] == 'done_fail':
			global_context['current_user_intents'] = ['done_fail']
			global_context['current_state'] = 'done'

		return user_input, global_context

	def step(self, agent_action, global_context, user_input):

		# print('Agent: {}'.format(agent_action))

		reward  = reward_function(agent_action, global_context)

		user_input, global_context = self.get_response_and_take_action(global_context, user_input)

		return user_input, reward, global_context, global_context['current_state']=='done'