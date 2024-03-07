import copy
import os
import openai

class Model():
    def __init__(self):
        pass

    def init_history(self):
        pass
    
    def query(self, x):
        pass

    def set_system_prompt(self, prompt):
        pass
    
    
class OPENAIModel(Model):
    def __init__(self, model_name, default_sleep=5):
        self.model_name = model_name
        self.history = []
        self.default_sleep = default_sleep
        self.max_retries = 3
        assert "OPENAI_KEY" in os.environ and os.environ['OPENAI_KEY'] != "", "Please export your openai key with environment variable OPENAI_KEY."
        openai.api_key = os.environ['OPENAI_KEY']
        self.usage = 0

    def init_history(self):
        self.history = []
    
    def set_system_prompt(self, prompt):
        self.history.append({"role": "system", "content": prompt})
    
    def append_user_message(self, x):
        self.history.append({"role": "user", "content": x})
        return self.history

    def append_system_response(self, x):
        self.history.append({"role": "assistant", "content": x})
        return self.history
    
    def get_history(self):
        return copy.deepcopy(self.history)

    def chatgpt(self,messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages = messages,
            temperature=0,
            max_tokens=4096,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
        return  response

    def gpt4(self, messages):
        # print(messages)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = messages,
            temperature=0,
            max_tokens=4096,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
        return  response
    
    def query(self, x):
        import time
        time.sleep(self.default_sleep)
        messages = self.append_user_message(x)
        
        if self.model_name == 'gpt-3.5':
            query_fn = self.chatgpt
        elif self.model_name == 'gpt-4':
            query_fn = self.gpt4

        response = ""
        tries = 0
        while (response == "" or  response == 'CONTENT_FILTER') and tries < self.max_retries:
            try:
                response = query_fn(messages)  
            except openai.error.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                response = "ERROR"
                pass
            except openai.error.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}")
                time.sleep(60)
                response = query_fn(messages)
                pass
            except openai.error.RateLimitError as e:
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(60)
                response = query_fn(messages)
                pass
            except:
                response = "ERROR"
                pass
            if response =="ERROR":
                response = "CONTENT_FILTER."
            else:
                self.usage += response['usage']['total_tokens']
                response = response["choices"][0]["message"]["content"]
        return response

class ClaudeModel(OPENAIModel):
    def __init__(self, model_name, default_sleep=1):
        from anthropic import Anthropic

        # assert "ANTHROPIC_API_KEY" in os.environ and os.environ['ANTHROPIC_API_KEY'] != "", "Please export your ANTHROPIC key with environment variable ANTHROPIC_API_KEY."
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic = Anthropic()

        self.model_name = model_name
        # self.client = anthropic.Client(os.environ['ANTHROPIC_KEY'])
        self.default_sleep = default_sleep
    
    def query(self, x):
        messages = self.append_user_message(x)
        # claude does not accept system prompt in messages API.
        if messages[0]['role'] == 'system':
            system, messages = messages[0], messages[1:]
            system_prompt = system['content']
        else:
            system_prompt = None
        
        import time
        time.sleep(self.default_sleep)
        n_retries = 5
        while n_retries > 0:
            try:
                completion = self.anthropic.completions.create(
                    model=self.model_name, 
                    system_prompt=system_prompt,
                    max_tokens_to_sample=1024,
                    messages=messages,
                    temperature=0,
                )
                n_retries = -1
            except Exception as e:
                print(e)
                n_retries -= 1
                time.sleep(self.default_sleep)
        if n_retries == 0:
            response = "ERROR"
        else:
            response = completion.completion
        return response

import boto3
import json

class ClaudeModel_bedrock(OPENAIModel):
    def __init__(self, default_sleep=1):

        # assert "ANTHROPIC_API_KEY" in os.environ and os.environ['ANTHROPIC_API_KEY'] != "", "Please export your ANTHROPIC key with environment variable ANTHROPIC_API_KEY."
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        self.bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.environ.get('aws_bedrock_region')
            )
        self.default_sleep = default_sleep
        print('model is claude')
    
    def query(self, x):
        modelId = 'anthropic.claude-v2'
        accept = 'application/json'
        contentType = 'application/json'

        messages = self.append_user_message(x)
        
        prompt = ''
        for i in range(len(messages)):
            if i == 0:
                continue
            elif i%2==1:
                prompt += 'Human: '+ messages[i]['content']+' \n\n'
            else:
                prompt += 'Assistant: '+ messages[i]['content']+' \n\n'
        
        prompt += 'Assistant: '
        body = json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 300,
            "temperature": 0.1,
            "top_p": 0.9,
        })
        import time
        time.sleep(self.default_sleep)
        n_retries = 5
        while n_retries > 0:
            try:
                response = self.bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
                n_retries = -1
            except Exception as e:
                print(e)
                n_retries -= 1
                time.sleep(self.default_sleep)
        if n_retries == 0:
            response = "ERROR"
        else:
            response_body = json.loads(response.get('body').read())
            response = response_body.get('completion')
        return response

class VLLMOENAIModel(OPENAIModel):
    def __init__(self, model_name, default_sleep=5):
        self.model_name = model_name
        self.history = []
        self.default_sleep = default_sleep
        openai_api_key = "EMPTY"
        from openai import OpenAI

        openai_api_base = "http://localhost:8000/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')
        # detect invalid model_max_length of llama-2
        if self.tokenizer.model_max_length > 10000000:
            self.tokenizer.model_max_length = 2048
        if 'alpaca-7b' in model_name:
            # in the original version, the max length is weirdly small. 
            self.tokenizer.model_max_length = 2048

        self.max_new_tokens = 256
        # approximate count
        self.token_per_turn = 5
        self.max_tokens = self.tokenizer.model_max_length
        # for single turn truncation
        self.tokenizer.model_max_length -= (self.token_per_turn+self.max_new_tokens)
    
    def truncate_messages(self, messages, max_len):
        tot = 0
        for i in range(len(messages)-1, -1, -1):
            cur_cnt = len(self.tokenizer.encode(messages[i]['content'], add_special_tokens=True, truncation=True))
            tot += cur_cnt
            tot += self.token_per_turn
            if tot > max_len:
                if messages[i]['role'] == 'user':
                    # the format have to be user/assistant/user/assistant ...
                    try:
                        assert i+2 < len(messages)
                    except:
                        breakpoint()
                    return messages[i+2:]
                else:
                    return messages[i+1:]
        return messages
    
    def query(self, x):
        messages = self.append_user_message(x)
        # mistral doesn't take system message
        if "Mistral" in self.model_name and messages[0]['role'] == 'system':
            messages = messages[1:]
            
        # truncate messages
        messages = self.truncate_messages(messages, max_len=self.tokenizer.model_max_length)
        assert len(messages) > 0
        
        import time
        time.sleep(self.default_sleep)
        n_retries = 5
        while n_retries > 0:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=messages,
                    temperature=0,
                    max_tokens=self.max_new_tokens,
                )
                n_retries = -1
            except Exception as e:
                print(e)
                n_retries -= 1
                time.sleep(self.default_sleep)
        if n_retries == 0:
            response = "ERROR"
        else:
            response = response.choices[0].message.content
        return response
    
class WithClsModelWrapper(object):
    def __init__(self, model, cls_model_path, use_post_prompt=False):
        self.model = model
        
        # load cls model
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.device = 'cuda'
        self.cls_tok = AutoTokenizer.from_pretrained(cls_model_path)
        self.cls = AutoModelForSequenceClassification.from_pretrained(cls_model_path).to(self.device)

        self.post_prompt = ""
        if use_post_prompt:
            self.post_prompt = "If you understand my requirements, please first repeat the requirement and fulfill the following task.\n"
    
    def check_relevant(self, query, instructions):
        dup_queries = [query, ] * len(instructions)
        instructions = [item['content'] for item in instructions]
        inputs = self.cls_tok(dup_queries, instructions, truncation=True, return_tensors='pt', padding=True).to(self.device)
        # TODO: setup batched inference
        logits = self.cls(**inputs).logits # [n, 2]
        output = logits.argmax(-1).cpu().tolist() # [n]
        return output
    
    def query(self, x):
        # for all historic queries, check whether they are relevant
        prev_instructions = [item for item in self.model.history if item['role']=='user']
        if prev_instructions:
            relevant_output = self.check_relevant(x, prev_instructions)
            relevant_contents = [prev_instructions[i]['content'] for i in range(len(prev_instructions)) if relevant_output[i] == 1]
            if relevant_contents:
                # based on the relevant output, construct the prompt
                p = "Given my previous instructions:\n"
                relevant_prompt = "{idx}. {instruct}\n"
                for idx, item in enumerate(relevant_contents):
                    p += relevant_prompt.format(idx=idx, instruct=item)
                x = p + x
                x = x + self.post_prompt
            else:
                p = ""
            response = self.model.query(x)
            # remove history
            self.model.history[-1]['content'] = self.model.history[-1]['content'][len(p):]
            return response
        else:
            # update x
            return self.model.query(x)
        
    def init_history(self):
        return self.model.init_history()
    
    def set_system_prompt(self, prompt):
        return self.model.set_system_prompt(prompt)
    
    def append_user_message(self, x):
        return self.model.append_user_message(x)

    def append_system_response(self, x):
        return self.model.append_system_response(x)
 
    def get_history(self):
        return self.model.get_history()

# if __name__ == "__main__":
#     # model_name = "/yanjianhao/huggingface/llama-2-7b-chat-hf"
#     # model_name = "/yanjianhao/huggingface/alpaca-7b"
#     model_name = "/yanjianhao/huggingface/Mistral-7B-Instruct-v0.2"
#     # from transformers import AutoTokenizer 
#     model = HFModel(model_name)
#     print(model.prompt_type)

#     # history = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Please translate the following sentence from English into Chinese.\n[English]: The hacked up version of Jedi Knight was crashing because it was calling a function off the end of a vtable. \n[Chinese]: "}, {"role": "assistant", "content": "被修改过的Jedi Knight版本因为调用了vtable末尾的函数而崩溃。\n\n[English]: The hacked up version of Jedi Knight was crashing due to calling a function off the end of a vtable.\n[Chinese]: 被修改过的Jedi Knight版本因为调用了vtable末尾的函数而崩溃了。\n\n[English]: The hacked up Jedi Knight was crashing because it called a function past the end of a vtable.\n["}, {"role": "user", "content": "In the following translations, you should translate function to 集会."}, {"role": "assistant", "content": "I apologize for the mistake in my previous response. Here's the corrected translation:\n\n[English]: The hacked up version of Jedi Knight was crashing because it was calling a function off the end of a vtable.\n[Chinese]: 被修改过的Jedi Knight版本因为调用了vtable末尾的函数而崩溃了。\n[Chinese with \"function\" translated to \"集会\" as requested]: 被修改过的Jedi Knight版本因为调用了vtable末尾的集会而�"}, {"role": "user", "content": "Please translate the following sentence from English into Chinese.\n[English]: The hacked up version of Jedi Knight was crashing because it was calling a function off the end of a vtable. \n[Chinese]: "}, {"role": "assistant", "content": "被修改过的Jedi Knight版本因为调用了vtable末尾的函数而崩溃了。\n\n[English]: The hacked up version of Jedi Knight was crashing because it called a function past the end of a vtable.\n[Chinese]: 被修改过的Jedi Knight版本因为调用了vtable之外的函数而崩溃了。\n\n[English]: The hacked up Jedi Knight was crashing because it called a function off the end of a vtable.\n"}]
#     # tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # prompt = tokenizer.apply_chat_template(history[1:], tokenize=False)
#     # print(prompt)
#     # prompt = model.apply_chat_template(history)
#     # print(prompt)
    