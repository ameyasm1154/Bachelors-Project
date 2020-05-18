from dialogue_config import all_intents, all_slots, user_sim_default_request
from utils import convert_list_to_dict
import numpy as np
import copy

class State_Tracker:

    def __init__(self, constants, global_context):

        self.match_key = user_sim_default_request
        self.intents_dict = convert_list_to_dict(all_intents)
        self.num_intents = len(all_intents)
        self.slots_dict = convert_list_to_dict(all_slots)
        self.num_slots = len(all_slots)
        self.max_round_num = constants['run']['max_round_num']
        self.none_state = np.zeros(self.get_state_size())
        self.reset()

    def get_state_size(self):

        return 2 * self.num_intents + 5 * self.num_slots

    def reset(self):

        self.history = []
        self.round_num = 0

    def print_history(self):

    	for action in self.history:
            print(action)
    
    def get_state(self, done=False):

        if done: return self.none_state

        user_action = self.history[-1]
        if len(self.history) > 1: 
            last_agent_action = self.history[-2] 
        else: 
            last_agent_action = self.history[-1]

        user_action_representation = np.zeros((self.num_intents,))
        for user_intent in user_action['current_user_intents']:
        	user_action_representation[self.intents_dict[user_intent]] = 1.0
        # print(user_action_representation)

        user_inform_slots_representation = np.zeros((self.num_slots,))
        for key in user_action['current_user_informs'].keys():
        	if user_action['current_user_informs'][key] != '':
        		user_inform_slots_representation[self.slots_dict[key]] = 1.0
        # print(user_inform_slots_representation)

        user_request_slots_representation = np.zeros((self.num_slots,))
        for key in user_action['current_user_requests'].keys():
        	if user_action['current_user_requests'][key] != '':
        		user_request_slots_representation[self.slots_dict[key]] = 1.0
        # print(user_request_slots_representation)

        agent_action_representation = np.zeros((self.num_intents,))
        for agent_intent in last_agent_action['current_agent_intents']:
        	agent_action_representation[self.intents_dict[agent_intent]] = 1.0
        # print(agent_action_representation)

        agent_inform_slots_representation = np.zeros((self.num_slots,))
        for key in last_agent_action['current_agent_informs'].keys():
        	if last_agent_action['current_agent_informs'][key] != '':
        		agent_inform_slots_representation[self.slots_dict[key]] = 1.0
        # print(agent_inform_slots_representation)

        agent_request_slots_representation = np.zeros((self.num_slots,))
        for key in last_agent_action['current_agent_requests'].keys():
        	if last_agent_action['current_agent_requests'][key] != '':
        		agent_request_slots_representation[self.slots_dict[key]] = 1.0
        # print(agent_request_slots_representation)

        all_inform_slots_representation = np.zeros((self.num_slots,))
        for key in self.history[-1]['constant_user_informs'].keys():
            if self.history[-1]['constant_user_informs'][key] != '':
                all_inform_slots_representation[self.slots_dict[key]] = 1.0
        # print(all_inform_slots_representation)

        state_representation = np.hstack(

        	[user_action_representation, user_inform_slots_representation, user_request_slots_representation,
        	agent_action_representation, agent_inform_slots_representation, agent_request_slots_representation,
        	all_inform_slots_representation]

        ).flatten()

        '''state_representation = np.hstack(

        	[user_action_representation, user_inform_slots_representation, user_request_slots_representation,
        	all_inform_slots_representation]

        ).flatten()'''

        return state_representation 

    def update_state_tracker_state(self, global_context):

        self.history.append(global_context)

        if global_context['controller'] == 'user':
            self.round_num += 1

        return global_context