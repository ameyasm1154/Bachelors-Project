from dqn_agent import DQN_Agent
from state_tracker import State_Tracker
import pickle, argparse, json, math
from utils import remove_empty_slots
from user import User
from user_goals import user_goals
from dialogue_config import new_global_context, agent_actions
from utils import reward_function
from matplotlib import pyplot as plt
import os

def run_round(state, global_context, user_input, dqn_agent, state_tracker, user, warmup=False):

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

	prev_global_context = global_context
	
	action_rewards = {}
	for curr_action in agent_actions:
		action_rewards[dqn_agent.map_action_to_index(curr_action)] = reward_function(curr_action, prev_global_context)
	
	agent_action = dqn_agent.map_index_to_action(max(action_rewards, key=action_rewards.get))
	
	global_context, next_agent_action = dqn_agent.take_action(agent_action, global_context)

	# print(next_agent_action)

	global_context = state_tracker.update_state_tracker_state(global_context)

	user_action, reward, global_context, done = user.step(next_agent_action, global_context, user_input)

	global_context = state_tracker.update_state_tracker_state(global_context)

	next_state = state_tracker.get_state(done)

	return next_state, reward, done, global_context, next_agent_action