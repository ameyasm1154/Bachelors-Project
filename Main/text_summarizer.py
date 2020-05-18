from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def get_summary(article):
    print('starting summarization...')
    article = article.strip().replace("\n", "")
    inputs = tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=512)
    outputs = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=False)

    ids = []
    for i in outputs:
        for j in i:
            ids.append(j)
    
    return tokenizer.decode(ids)