from dialogue_config import FAIL, SUCCESS, REWARD, PENALTY, NO_OUTCOME

def reward_function(action, global_context):

	reward_value = 0.0

	if action['intent'] == 'make_api_call':
		all_context_available = True
		for key in global_context['constant_user_informs'].keys():
			all_context_available &= (global_context['constant_user_informs'][key] != '')
		if all_context_available:
			reward_value += 5 * REWARD
		elif not all_context_available:
			reward_value += 5 * PENALTY		

	elif action['intent'] == 'request':
		for key in action['request_slots'].keys():
			if action['request_slots'][key] != '' and global_context['constant_user_informs'][key] == '':
				reward_value += 5 * REWARD
			elif action['request_slots'][key] != '' and global_context['constant_user_informs'][key] != '':
				reward_value += 5 * PENALTY
		for key in global_context['constant_user_informs'].keys():
			if global_context['constant_user_informs'][key] == '' and  action['request_slots'][key] == '':
				reward_value += 5 * PENALTY

	elif action['intent'] == 'greet':
		all_context_unavailable = True
		for key in global_context['constant_user_informs'].keys():
			all_context_unavailable &= (global_context['constant_user_informs'][key] == '')
		if all_context_unavailable:
			reward_value += 5 * REWARD
		elif not all_context_unavailable:
			reward_value +=  5 * PENALTY

	elif action['intent'] == 'inform':
		for key in action['inform_slots'].keys():
			if action['inform_slots'][key] != '' and global_context['current_user_requests'][key] != '':
				reward_value += 5 * REWARD
				all_context_available = True
				for key in global_context['constant_user_informs'].keys():
					all_context_available &= (global_context['constant_user_informs'][key] != '')
				if all_context_available:	
					reward_value += 5 * REWARD
				elif not all_context_available:
					reward_value += 5 * PENALTY
			elif action['inform_slots'][key] == '' and global_context['current_user_requests'][key] != '':
				reward_value += 5 * PENALTY

	'''if global_context['current_state'] == 'done':
		if 'done_success' in global_context['current_user_intents']:
			reward_value += SUCCESS
		elif 'done_fail' in global_context['current_user_intents']:
			reward_value += FAIL
		elif global_context['current_round_num'] == constants['run']['max_round_num']:
			reward_value += FAIL'''

	return reward_value

def convert_list_to_dict(lst):

    if len(lst) > len(set(lst)):
        raise ValueError('List must be unique!')

    return {k: v for v, k in enumerate(lst)}


def remove_empty_slots(dic):

    for id in list(dic.keys()):
        for key in list(dic[id].keys()):
            if dic[id][key] == '':
                dic[id].pop(key)