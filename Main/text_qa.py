import torch
from transformers import AlbertTokenizer, AlbertForQuestionAnswering
from tfidf import *

text_qa_tokenizer = AlbertTokenizer.from_pretrained('ahotrod/albert_xxlargev1_squad2_512')
text_qa_model = AlbertForQuestionAnswering.from_pretrained('ahotrod/albert_xxlargev1_squad2_512')

def get_answer_for_text(question, text):

    paras = text.split('.')
    ranked_paras =  get_similarity([question], paras)
    selected_para = paras[ranked_paras[0][0]]

    input_dict = text_qa_tokenizer.encode_plus(question, selected_para, return_tensors='pt', max_length=512)
    input_ids = input_dict["input_ids"].tolist()
    start_scores, end_scores = text_qa_model(**input_dict)

    start = torch.argmax(start_scores)
    end = torch.argmax(end_scores)

    all_tokens = text_qa_tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = ''.join(all_tokens[start: end + 1]).replace('▁', ' ').strip()
    answer = answer.replace('[SEP]', '')

    return answer if answer != '[CLS]' and len(answer) != 0 else 'could not find an answer'