from collections import NamedTuple
import json
import jsonlines
import litellm
import openai
import os

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
        if os.path.exists(self.cache_file):
            self.cache = dict([tuple(row) for row in self.cache_file])
            self.cache_file = jsonlines.open(self.cache_filename, "a", flush=True)
        else:
            self.cache = {}
            self.cache_file = jsonlines.open(self.cache_filename, "w", flush=True)


    def respond(self, prompt: str) -> litellm.ModelResponse:
        # from litellm import completion
        # response = completion(model="gpt-3.5-turbo", messages=messages)
        if prompt in self.cache:
            response_dict = self.cache[prompt]
            response = litellm.ModelResponse.parse_raw(json.dumps((response_dict.json())))
        else:
            breakpoint()
            response = openai.ChatCompletion.create(
                model=self.name,
                messages=[{"role": "user", "content": prompt}]
            )
            response_dict = response.choices[0]

            response = litellm.ModelResponse(
                choices=[litellm.Choices(message=litellm.Message(response_dict.message.role, response_dict.message.content))]
            )
            self.cache[prompt] = response.json()
            self.cache_file.write(response.json())
        return response