# user_intents = ['open_file', 'image_Q&A', 'document_Q&A', 'image_segmentation', 'diagnosis', 'done_success', 'done_fail']

new_global_context = {

	'controller': '',
	
	'current_state': '', # possible states -> grounded, slot_filling, context_switch, initiative, done

	'constant_user_informs': {'instruction':'', 'task_name':'', 'context_name':''},

    'constant_user_requests': {'answer':''},

    'all_user_informs': [{'instruction':'', 'task_name':'', 'context_name':''}],

	'all_user_requests': [{'answer':''}],

	'all_agent_informs': [{'answer':''}],

	'all_agent_requests': [{'instruction':'', 'task_name':'', 'context_name':''}],

	'current_user_informs': {'instruction':'', 'task_name':'', 'context_name':''},

	'current_user_requests': {'answer':''},

	'current_agent_requests': {'instruction':'', 'task_name':'', 'context_name':''},

	'current_agent_informs': {'answer':''},

	'current_user_intents': [],

	'current_user_entities': [],

	'current_agent_intents': [],

	'current_agent_entities': [],

	'current_round_num': 0
}

user_intents = ['inform', 'request', 'done_success', 'done_fail']

user_sim_default_request = 'answer'

user_sim_default_inform = 'task_name'

agent_intents = ['make_api_call', 'inform', 'request', 'greet']

agent_inform_slots = ['answer']

agent_request_slots =  ['instruction', 'task_name', 'context_name']

# default value of 'answer' -> 'PLACEHOLDER'
# default value of 'instruction'/'task_name'/'context_name' -> 'UNK'

agent_actions = [

	{'intent': 'make_api_call', 
	 'request_slots': {'instruction':'', 'task_name':'', 'context_name':''}, 
	 'inform_slots': {'answer': ''}},

	{'intent': 'request', 
	 'request_slots': {'instruction':'UNK', 'task_name':'', 'context_name':''}, 
	 'inform_slots': {'answer': ''}},

	{'intent': 'request', 
	 'request_slots': {'instruction':'', 'task_name':'UNK', 'context_name':''}, 
	 'inform_slots': {'answer': ''}},

	{'intent': 'request', 
	 'request_slots': {'instruction':'', 'task_name':'', 'context_name':'UNK'}, 
	 'inform_slots': {'answer': ''}},

	{'intent': 'request', 
	 'request_slots': {'instruction':'UNK', 'task_name':'UNK', 'context_name':''}, 
	 'inform_slots': {'answer': ''}},

	{'intent': 'request', 
	 'request_slots': {'instruction':'UNK', 'task_name':'', 'context_name':'UNK'}, 
	 'inform_slots': {'answer': ''}},

	{'intent': 'request', 
	 'request_slots': {'instruction':'', 'task_name':'UNK', 'context_name':'UNK'}, 
	 'inform_slots': {'answer': ''}},

	{'intent': 'inform', 
	 'request_slots': {'instruction':'', 'task_name':'', 'context_name':''}, 
	 'inform_slots': {'answer': 'PLACEHOLDER'}},

	{'intent': 'greet', 
	 'request_slots': {'instruction':'UNK', 'task_name':'UNK', 'context_name':'UNK'}, 
	 'inform_slots': {'answer': ''}}

]

FAIL = -1
NO_OUTCOME = 0
SUCCESS = 1
REWARD = 1
PENALTY = -1

all_intents = ['inform', 'request', 'done_success', 'done_fail', 'make_api_call', 'greet']
all_slots = ['instruction', 'task_name', 'context_name', 'answer']