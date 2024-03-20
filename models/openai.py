import json
import jsonlines
import litellm
from openai import OpenAI
import os
from typing import Optional
import uuid

class Message():
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content
        }

class Response:
    def __init__(self, message):
        self.message = message

class OpenAIModel:
    def __init__(self, name):
        assert name in ["gpt-3.5-turbo", "gpt-4"]
        self.name = name
        self.cache_filename = f"{name}_cache.jsonl"
        if os.path.exists(self.cache_filename):
            self.cache = dict([tuple(row) for row in  jsonlines.open(self.cache_filename, "r")])
            self.cache_file = jsonlines.open(self.cache_filename, "a", flush=True)
        else:
            self.cache = {}
            self.cache_file = jsonlines.open(self.cache_filename, "w", flush=True)


    def respond(self, prompt: str, constraint_type: Optional[str] = None) -> litellm.ModelResponse:
        # from litellm import completion
        # response = completion(model="gpt-3.5-turbo", messages=messages)
        if prompt in self.cache:
            response_dict = self.cache[prompt]
            response = litellm.ModelResponse.parse_raw(json.dumps(response_dict))
        else:
            '''
            constraint_types = ['content', 'situation', 'style', 'format', 'example', 'mixed']
            for constraint_type in constraint_types:
                path = f"/Users/vijay/Documents/code/compose-instruct/data/FollowBench/api_output/gpt_3_5_turbo_{constraint_type}_constraint.jsonl"
                for row in jsonlines.open(path, "r"):
                    obj = {'id': "chatcmpl-" + str(uuid.uuid4()), 'choices': row["choices"]}
                    cache_row = [row["prompt_new"], obj]
                    self.cache_file.write(cache_row)
            '''


            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.name,
                messages=[{"role": "user", "content": prompt}]
            )

            response_dict = response.choices[0]

            message = litellm.Message(role=response_dict.message.role, content=response_dict.message.content, function_call=None, tool_calls=None)
            litellm_response = litellm.ModelResponse()
            litellm_response.choices = litellm.Choices(message=message)
            cache_record = [prompt, litellm_response.json()]
            self.cache[prompt] = cache_record
            self.cache_file.write(cache_record)
        return response