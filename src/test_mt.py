import fire
import os
import jsonlines
import string
import random
import json
import tqdm
import jieba
import spacy

from model_utils import OPENAIModel, VLLMOENAIModel, ClaudeModel_bedrock, WithClsModelWrapper

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

def read_json(file_path):
    # Open the JSON Lines file for reading
    with jsonlines.open(file_path) as reader:
        # Iterate over each line in the file
        return [line for line in reader]
    
def get_word_list_he(s):
    import hebrew_tokenizer as ht
    tokens = ht.tokenize(s)
    return [token[1] for token in tokens]

def get_word_list_zh(s):
    import jieba
    tokens = jieba.cut(s)
    return tokens
    
def get_word_list_en_old(s):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    word_list = s.translate(translator).split()
    return word_list

def get_word_list_en_new(s):
    global nlp
    doc = nlp(s)
    word_list = [token.text for token in doc]
    return word_list

def get_word_list(s, lang):
    if len(s) == 0:
        return []
    if lang == 'en':
        return get_word_list_en_new(s)
    elif lang == 'zh':
        return get_word_list_zh(s)
    elif lang == 'he':
        return get_word_list_he(s)
    else:
        raise NotImplementedError()
    
def main(
    feedback_file,
    output_file,
    tgt_lang,
    model_name='llama-2-7b',
    feedback_keyword='translation',
    src_lang='en',
    debug=False,
    strategy='standard',
    cls_model_path="",
):
    if not tgt_lang:
        raise
    # read file
    data = read_json(feedback_file)

    # init model 
    if model_name in ['gpt-3.5', 'gpt-4']:
        model = OPENAIModel(model_name, default_sleep=0.5)
    elif model_name == 'claude':
        # print('model is claude')
        model = ClaudeModel_bedrock(default_sleep=0.5)
    else:
        model = VLLMOENAIModel(model_name, default_sleep=0.1)

    if strategy.startswith('cls'):
        assert cls_model_path != ""
        if 'repeat' in strategy:
            model = WithClsModelWrapper(model, cls_model_path=cls_model_path, use_post_prompt=True)
        else:
            model = WithClsModelWrapper(model, cls_model_path=cls_model_path)
    
    # Check the current number of lines in the output_file to resume correctly
    start_index = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf8') as fo:
            for start_index, _ in enumerate(fo, 1):
                pass
    
    print(model)

    fo = open(output_file, 'a', encoding='utf8')
    # truncate data
    data = data[start_index:]
    # iterative query
    for i, data_item in tqdm.tqdm(enumerate(data), initial=start_index):

        if debug:
            if i > 30: break
            if i < 14:
                continue
        queries = data_item['queries']
        feedbacks = data_item['feedbacks']
        system_prompt = data_item['system_prompt']
        feedback_prompt = data_item['feedback_prompt']
        feedback_mapping_prompt = data_item['feedback_mapping_prompt']
        
        # init history
        model.init_history()
        # set system prompt
        model.set_system_prompt(system_prompt)
        
        prev_response = ""
        prev_query = ""
        for query in queries:
            response = ""
            conflict_flag = False
            # maybe feedback
            if query == '[MAYBE FEEDBACK]':
                # assert prev_response != ""
                query = ""
                used_feedbacks = []
                for src in feedbacks:
                    # in word list
                    if src in get_word_list(prev_query, lang=src_lang):
                        tgts = feedbacks[src][feedback_keyword]
                        # forced feedback
                        if isinstance(tgts, str):
                            choose_tgt = tgts
                        # feedback from a list of choices
                        else:
                            assert isinstance(tgts, list)
                            # not answered tgts
                            na_tgts = [tgt for tgt in tgts if tgt not in get_word_list(prev_response, lang=tgt_lang)]
                            try:
                                # assert len(na_tgts) >= len(tgts)-1
                                assert len(na_tgts) >= 1
                            except:
                                conflict_flag = True
                                choose_tgt = "<EMPTY>"
                                break
                            # try:
                            choose_tgt = random.choice(na_tgts)
                        query += feedback_mapping_prompt.format(src=src, tgt=choose_tgt)
                        used_feedbacks.append(src)
                if query != "":
                    query = feedback_prompt + query
                # remove used feedbacks
                for src in used_feedbacks:
                    feedbacks.pop(src)
            
            if conflict_flag is True:
                model.append_user_message('<CONFLICT>')
                break
                        
            if len(query) > 0:
                response = model.query(query)
                usage = getattr(model, 'usage', None)
                if usage:
                    print(f'Current Usage: {usage}')
                model.append_system_response(response)
            prev_response = response
            prev_query = query
        # save results
        fo.write(json.dumps(model.get_history(), ensure_ascii=False) + '\n')
        fo.flush()

if __name__ == '__main__':
    fire.Fire()