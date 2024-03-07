import fire
import os, copy
import jsonlines
import string
import random
import json
import tqdm
import jieba
import os

from src.model_utils import OPENAIModel, VLLMOENAIModel, ClaudeModel_bedrock
from src.instructions_checker import TitleChecker, GreetingsChecker, ParagraphChecker, ResponseLanguageChecker, SignatureChecker, SentenceChecker

def read_json(file_path):
    # Open the JSON Lines file for reading
    with jsonlines.open(file_path) as reader:
        # Iterate over each line in the file
        return [line for line in reader]
    
def get_word_list_zh(s):
    tokens = jieba.cut(s)
    return tokens
    
def get_word_list_en(s):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    word_list = s.translate(translator).split()
    return word_list

def get_word_list(s, lang):
    if len(s) == 0:
        return []
    if lang == 'en':
        return get_word_list_en(s)
    elif lang == 'zh':
        return get_word_list_zh(s)
    else:
        raise NotImplementedError()

def get_domain_from_feedback_prompt(p):
    if p.startswith("When writing my work emails"):
        return 'work'
    elif p.startswith("When writing my emails related to schools"):
        return 'school'
    elif p.startswith("When writing my emails to my family"):
        return 'family'
    elif p.startswith("When writing my emails to my friends"):
        return "friend"
    else:
        raise ValueError()

def select_checker(check_str):
    if check_str == 'greetings': return GreetingsChecker()
    elif check_str == 'signature': return SignatureChecker()
    elif check_str == 'title': return TitleChecker()
    elif check_str == 'paragraph': return ParagraphChecker()
    elif check_str == 'response_language': return ResponseLanguageChecker()
    elif check_str == 'sentence': return SentenceChecker()
    else: 
        raise ValueError()
    
def main(
    feedback_file,
    output_file,
    model_name='llama-2-7b',
    debug=False
):
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
        # model = HFModel(model_name)
    
    # Check the current number of lines in the output_file to resume correctly
    run_ids = []
    unfinished = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf8') as fo:
            lines = fo.readlines()
        # check CONTENT_FILTER
        for idx, line in enumerate(lines):
            if "CONTENT_FILTER" in line:
                run_ids.append(idx)
        # queries not finished by current output
        unfinished = len(lines)
    run_ids += list(range(unfinished, len(data)))
    
    print(model)

    start_index = 0
    # setup start index
    if os.path.exists(output_file+'.tmp'):
        with open(output_file+'.tmp', 'r', encoding='utf8') as fo:
            for start_index, _ in enumerate(fo, 1):
                pass
    fo_temp = open(output_file+'.tmp', 'a', encoding='utf8')
    # run ids
    cur_run_ids = run_ids[start_index:]
    cur_run_ids = set(cur_run_ids)
    # iterative query
    for i, data_item in tqdm.tqdm(enumerate(data)):
        if i not in cur_run_ids:
            continue

        if debug:
            if i > 20: break
        queries = data_item['queries']
        feedbacks = data_item['feedbacks']
        system_prompt = data_item['system_prompt']
        feedback_prompts = data_item['feedback_prompts']
        
        # init history
        model.init_history()
        # set system prompt
        model.set_system_prompt(system_prompt)
        
        # we need to append more info in saved json
        history_to_save = model.get_history()
        
        prev_response = ""
        fb_idx = 0
        for query in queries:
            response = ""
            checker = None
            # maybe feedback
            if query == '[MAYBE FEEDBACK]':
                # assert prev_response != ""
                query = ""
                checker_str = feedbacks[fb_idx]['checker']
                checker = select_checker(checker_str)
                # init checker
                check_all_choices = [checker.check_following(prev_response, choice) for choice in feedbacks[fb_idx]['choices']]
                
                # select from instructions that are not fulfilled yet.
                available_choices = [choice for i, choice in enumerate(feedbacks[fb_idx]['choices']) if check_all_choices[i] is False]
                    
                feedback_prompt = feedback_prompts[fb_idx]
                if len(available_choices) > 0:
                    fb_choice = random.choice(available_choices)
                    query = feedback_prompt.format(choice=fb_choice)
                else:
                    fb_choice = ""
                fb_idx += 1
                
            if len(query) > 0:
                response = model.query(query)
                usage = getattr(model, 'usage', None)
                if usage:
                    print(f'Current Usage: {usage}')
                    
                model.append_system_response(response)
                q, r = copy.deepcopy(model.history[-2]), copy.deepcopy(model.history[-1])
                assert q['role'] == 'user' and r['role'] == 'assistant'
                # if current query is the feedback
                if checker is not None:
                    q['type'] = 'feedback'
                    q['checker'] = checker_str
                    q['choice'] = fb_choice
                else:
                    q['type'] = 'normal'
                q['domain'] = get_domain_from_feedback_prompt(feedback_prompts[0])
                history_to_save.append(q)
                history_to_save.append(r)
            prev_response = response
        # save results
        fo_temp.write(json.dumps(history_to_save, ensure_ascii=False) + '\n')
        fo_temp.flush()
    
    # combine fo_temp and fo
    with open(output_file+'.tmp', 'r', encoding='utf8') as f:
        temp_lines = f.readlines()

    assert len(temp_lines) == len(run_ids)
    j = 0
    with open(output_file, 'w', encoding='utf8') as fo:
        for ii in range(len(data)):
            if j < len(run_ids) and ii == run_ids[j]:
                fo.write(temp_lines[j])
                j += 1
            else:
                fo.write(lines[ii])
            
    os.remove(output_file+'.tmp')
        

if __name__ == '__main__':
    fire.Fire()