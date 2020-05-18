from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
import tensorflow as tf
import random, copy
import numpy as np
import pickle
import json
import re
from dialogue_config import agent_actions
from utils import reward_function
from disease_dict import *
from text_qa import *
from visual_qa import *
from text_summarizer import *
from anomaly_detection import *

class DQN_Agent:

    def __init__(self, state_size, constants, global_context):

        self.C = constants['agent']
        self.memory = []
        self.memory_index = 0
        self.max_memory_size = self.C['max_mem_size']
        self.eps = self.C['epsilon_init']
        self.vanilla = self.C['vanilla']
        self.lr = self.C['learning_rate']
        self.gamma = self.C['gamma']
        self.batch_size = self.C['batch_size']
        self.hidden_size = self.C['dqn_hidden_size']

        self.load_weights_file_path = self.C['load_weights_file_path']
        self.save_weights_file_path = self.C['save_weights_file_path']

        if self.max_memory_size < self.batch_size:
            raise ValueError('Max memory size must be at least as great as batch size!')

        self.state_size = state_size
        self.possible_actions = agent_actions
        self.num_actions = len(self.possible_actions)

        self.rule_request_set = ['task_name', 'context_name', 'instruction']

        self.beh_model = self.build_model()

        self.diagnosis_model = pickle.load(open('diagnosis_model', 'rb'))

        # self._load_weights()

        self.reset()

    def build_model(self):

        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.hidden_size, activation='relu'))
        model.add(Dense(self.hidden_size, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))

        return model

    def reset(self):

        self.rule_current_slot_index = 0
        self.rule_requests = ['context_name', 'task_name', 'instruction']
        self.rule_phase = 'not done'

    def get_action(self, state, use_rule=False):

        if (self.eps > random.random()) and False:
            index = random.randint(0, self.num_actions - 1)
            action = self.map_index_to_action(index)
            return index, action
        else:
            if use_rule:
                return self.rule_action()
            else:
                return self.dqn_action(state)

    def map_action_to_index(self, response):

        for (i, action) in enumerate(self.possible_actions):
            if response == action:
                return i

        raise ValueError('Response: {} not found in possible actions'.format(response))

    def rule_action(self):

        if self.rule_current_slot_index < len(self.rule_requests):
            slot = self.rule_requests[self.rule_current_slot_index]
            self.rule_current_slot_index += 1
            rule_response = {'intent': 'request', 'request_slots': {'instruction': '', 'task_name': '', 'context_name': ''}, 'inform_slots': {'answer':''}}
            rule_response['request_slots'][slot] = 'UNK'

        elif self.rule_phase == 'not done':
            rule_response = {'intent': 'make_api_call', 'request_slots': {'instruction': '', 'task_name': '', 'context_name': ''}, 'inform_slots': {'answer':''}}
            self.rule_phase = 'done'

        elif self.rule_phase == 'done':
            rule_response = {'intent': 'inform', 'request_slots': {'instruction': '', 'task_name': '', 'context_name': ''}, 'inform_slots': {'answer': 'PLACEHOLDER'}}

        else:
            raise Exception('Should not have reached this clause')

        index = self.map_action_to_index(rule_response)
        return index, rule_response

    def dqn_action(self, state):

        index = np.argmax(self.dqn_predict_one(state))
        action = self.map_index_to_action(index)

        return index, action

    def map_index_to_action(self, index):

        for (i, action) in enumerate(self.possible_actions):
            if index == i:
                return copy.deepcopy(action)
        raise ValueError('Index: {} not in range of possible actions'.format(index))

    def dqn_predict_one(self, state):

        return self.dqn_predict(state.reshape(1, self.state_size)).flatten()

    def dqn_predict(self, states):

        preds = self.beh_model.predict(states)

        return preds

    def add_experience(self, state, action, reward, next_state, global_context, done):

        if len(self.memory) < self.max_memory_size:
            self.memory.append(None)
        self.memory[self.memory_index] = (state, action, reward, next_state, global_context, done)
        self.memory_index = (self.memory_index + 1) % self.max_memory_size

    def empty_memory(self):

        self.memory = []
        self.memory_index = 0

    def is_memory_full(self):

        return len(self.memory) == self.max_memory_size

    def take_action(self, action, global_context):

        global_context['controller'] = 'agent'

        if action['intent'] == 'greet':

            next_action = action

            for key in global_context['current_agent_requests'].keys(): 
                global_context['current_agent_requests'][key] = 'UNK'

            global_context['all_agent_requests'].append(action['request_slots'])

            return global_context, next_action

        if action['intent'] == 'request':

            next_action = action

            for key in global_context['current_agent_requests'].keys():
                if key in action['request_slots'].keys():
                    global_context['current_agent_requests'][key] = 'UNK'

            global_context['all_agent_requests'].append(action['request_slots'])

            return global_context, next_action

        if action['intent'] == 'inform':

            next_action = action

            for key in global_context['current_agent_informs'].keys():
                if key in action['inform_slots'].keys():
                    global_context['current_agent_informs'][key] = action['inform_slots'][key]

            global_context['all_agent_informs'].append(action['inform_slots'])

            return global_context, next_action

        if action['intent'] == 'make_api_call':

            next_action = action
            
            next_action, global_context = self.make_api_call(global_context)

            global_context, next_action = self.take_action(next_action, global_context)

            return global_context, next_action

    def make_api_call(self, global_context):

        # print('api call made according to -')
        # print('all informs -> ', global_context['all_user_informs'])
        # print('intents -> ', global_context['current_user_intents'])
        # print('entities -> ', global_context['current_user_entities'])
        # print(global_context)
        if global_context['constant_user_informs']['task_name'] == 'diagnosis_patterns':

            input_disease_vector = np.zeros(len(symptoms_dict))

            given_symptoms = []
            for symptom in global_context['current_user_entities'][-1]['symptoms']:
                given_symptoms.append(symptoms_dict[symptom])

            input_disease_vector[given_symptoms] = 1

            predicted_diagnosis = self.diagnosis_model.predict([input_disease_vector])[0]

            predicted_diagnosis_confidence = np.max(self.diagnosis_model.predict_proba([input_disease_vector])[0])

            answer = {}
            answer['diagnosis'] = predicted_diagnosis
            answer['confidence'] = predicted_diagnosis_confidence
            answer = json.dumps(answer)

            next_action = {'intent': 'inform', 'request_slots': {'instruction': '', 'task_name': '', 'context_name': ''}, 'inform_slots': {'answer': answer}}
        
            return next_action, global_context

        elif (global_context['constant_user_informs']['task_name'] == 'qna_intent_patterns') and (global_context['constant_user_informs']['context_name'][-4:] in ['.txt']):

            text_qa_filename = global_context['constant_user_informs']['context_name']
            f = open('static/texts/'+text_qa_filename, "r")
            text = str(f.read())

            question = global_context['constant_user_informs']['instruction']
            answer = get_answer_for_text(question, text)

            next_action = {'intent': 'inform', 'request_slots': {'instruction': '', 'task_name': '', 'context_name': ''}, 'inform_slots': {'answer': answer}}

            return next_action, global_context

        elif (global_context['constant_user_informs']['task_name'] == 'qna_intent_patterns') and (global_context['constant_user_informs']['context_name'][-4:] in ['.jpg', '.png']):

            clear_session()

            image_qa_filename = global_context['constant_user_informs']['context_name']

            question = global_context['constant_user_informs']['instruction']
            answer = get_answer_for_image(str(question), str(image_qa_filename))

            next_action = {'intent': 'inform', 'request_slots': {'instruction': '', 'task_name': '', 'context_name': ''}, 'inform_slots': {'answer': answer}}

            return next_action, global_context

        elif (global_context['constant_user_informs']['task_name'] == 'text_summarization_patterns') and (global_context['constant_user_informs']['context_name'][-4:] in ['.txt']):

            for_summary_filename = global_context['constant_user_informs']['context_name']
            f = open('static/texts/'+for_summary_filename, "r")
            text = str(f.read())

            summary = get_summary(text)

            next_action = {'intent': 'inform', 'request_slots': {'instruction': '', 'task_name': '', 'context_name': ''}, 'inform_slots': {'answer': summary}}

            return next_action, global_context

        elif (global_context['constant_user_informs']['task_name'] == 'anomaly_detection_patterns') and (global_context['constant_user_informs']['context_name'][-4:] in ['.jpg', '.png']):

            image_anomaly_detection_filename = global_context['constant_user_informs']['context_name']
            
            answer = detect_anomaly(image_anomaly_detection_filename)

            # print(answer)

            next_action = {'intent': 'inform', 'request_slots': {'instruction': '', 'task_name': '', 'context_name': ''}, 'inform_slots': {'answer': answer}}

            return next_action, global_context

        next_action = {'intent': 'inform', 'request_slots': {'instruction': '', 'task_name': '', 'context_name': ''}, 'inform_slots': {'answer': 'PLACEHOLDER'}}
        
        return next_action, global_context

    def train(self, global_context):

        num_batches = len(self.memory) // self.batch_size
        for b in range(num_batches):
            batch = random.sample(self.memory, self.batch_size)

            states = np.array([sample[0] for sample in batch])
            next_states = np.array([sample[3] for sample in batch])

            assert states.shape == (self.batch_size, self.state_size), 'States Shape: {}'.format(states.shape)
            assert next_states.shape == states.shape

            beh_state_preds = self.dqn_predict(states)
            if not self.vanilla:
                beh_next_states_preds = self.dqn_predict(next_states)  # For indexing for DDQN

            inputs = np.zeros((self.batch_size, self.state_size))
            targets = np.zeros((self.batch_size, self.num_actions))

            for i, (s, a, r, s, g, d) in enumerate(batch):
                # print('reward: {}'.format(r))
                t = beh_state_preds[i]
                for itr in range(len(t)):
                    t[itr] = reward_function(self.map_index_to_action(itr), g)
                # print(t)
                if not self.vanilla:
                    t[a] = r # + self.gamma * tar_next_state_preds[i][np.argmax(beh_next_states_preds[i])] * (not d)
                else:
                    t[a] = r # + self.gamma * np.amax(tar_next_state_preds[i]) * (not d)

                inputs[i] = s
                targets[i] = t
                # print('Targets: {}'.format(targets))

            self.beh_model.fit(inputs, targets, epochs=10, verbose=0)
            # input()

    def save_weights(self):

        if not self.save_weights_file_path:
            return
        save_file_path = 'dialogue_manager_dqn_model.h5'
        self.beh_model.save_weights(save_file_path)

    def _load_weights(self):

        if not self.load_weights_file_path:
            return
        load_file_path = 'dialogue_manager_dqn_model.h5'
        self.beh_model.load_weights(load_file_path)