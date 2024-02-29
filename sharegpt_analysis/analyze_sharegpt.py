import asyncio
from collections import Counter
import csv
import matplotlib.pyplot as plt
import openai
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import tiktoken
from sklearn.metrics import silhouette_score
from tqdm import tqdm


from prompt2model.utils import APIAgent
import json

def postprocess_response(response_text):
    if response_text.strip() == "NO DECOMPOSITION":
        return 0
    else:
        DEFAULT_RESPONSE = -99999
        try:
            if not response_text.startswith("["):
                return DEFAULT_RESPONSE
            counter = 1
            substring_found = False
            for i, c in enumerate(response_text[1:]):
                if c == "[":
                    counter += 1
                elif c == "]":
                    counter -= 1
                if counter == 0:
                    substring_found = True
                    break
            if substring_found:
                response_dict = json.loads(response_text[:i+2])
                return response_dict
            else:
                print(f"No JSON substring found in response: {response_text}")
                return DEFAULT_RESPONSE
        except:
            print(f"Error parsing response: {response_text}")
            return DEFAULT_RESPONSE

def query_gpt(prompts, tag):
    agent = APIAgent(model_name="gpt-3.5-turbo")

    cache_file = f"/tmp/azure_{tag}_cache.pkl"
    if os.path.exists(cache_file):
        responses = pickle.load(open(cache_file, "rb"))
    else:
        loop = asyncio.get_event_loop()
        responses = loop.run_until_complete(
            agent.generate_batch_completion(
                prompts,
                temperature=0,
                responses_per_request = 1,
                requests_per_minute = 100,
                token_buffer = 1000,
            )
        )
        pickle.dump(responses, open(cache_file, "wb"))

    postprocessed_tags = []
    for response in responses:
        tag_dict = postprocess_response(response["choices"][0]["message"]["content"])
        postprocessed_tags.append(tag_dict)
    return postprocessed_tags

def construct_skills_prompt(instruction):
    prompt = f"""I am trying to analyze the skills required by human instructions provided to language models such as yourself. These may be things like "numerical reasoning", "translation from English", "translation to Japanese", etc. Format your list of skills as a JSON string. Keep the list very short (1-4 skills), very high-level (e.g. "numerical reasoning" instead of "addition") and conceptually unique. There should almost always be at least one skill.

Examples
--------

- Instruction:
Let's play tic tac toe
- Skills:
["Create interactive game"]
-----
- Instruction:
Tell me the number of sentences below that are about dogs:

நான் நாய்க்கு சில முட்டைகளைக் கொடுத்தேன்
நான் பறவைக்கு கொஞ்சம் சோளம் கொடுத்தேன்
புலி பொமரேனியனைப் பார்த்தது
- Skills:
["Understand Tamil", "Classify topics", "Count sentences"]
-----
- Instruction:
Interpret the regression statistics below:
Regression Statistics 
Multiple R 0.983840293 
R Square 0.967941722 
Adjusted R Square 0.958782213 
Standard Error 229390.3205 
Observations 10 
 
ANOVA 
 df SS MS F Significance F 
Regression 2 1.11213E+13 5.56067E+12 105.6761684 5.89917E-06 
Residual 7 3.68339E+11 52619919134 
Total 9 1.14897E+13 
 
 Coefficients Standard Error t Stat P-value Lower 95% Upper 95% Lower 95.0% Upper 95.0%
Intercept 310055.0534 483971.7741 0.640646976 0.542147832 -834356.3408 1454466.448 -834356.3408 1454466.448
Employee Count -60461.95689 19894.68211 -3.039101432 0.018870807 -107505.4047 -13418.5091 -107505.4047 -13418.5091
Store Square Footage 91.18363601 17.59860205 5.181299955 0.001279055 49.56955481 132.7977172 49.56955481 132.7977172
- Skills:
["Explain summary statistics"]
-----
- Instruction:
i need apps script to search a place name in a column, and store the lat and long on other columns
- Skills:
["Write Google Apps Script code"]
-----
- Instruction:
which journal is similar to communications medicine?
- Skills:
["Describe the medical journal", "Identify similar journals"]
-----
- Instruction:
{instruction}
- Skills:
"""
    return prompt


def construct_requirements_prompt(instruction):
    prompt = f"""I am trying to analyze the requirements specified by human instructions provided to language models such as yourself. Requirements can be constraints (e.g. "the response is in Japanese") or preferences ("the response should use casual words whenever possible"). If there are no obvious requirements specified, return an empty list. Format your list of requirements as a JSON string.

Examples
--------

- Instruction:
Write a Japanese poem about robots that is at least 10 lines long and contains internal rhyme.
- Requirements:
["Response is a poem", "Response is in Japanese", "Response is about robots", "Response is at least 10 lines long", "Response contains internal rhyme"]
-----
- Instruction:
Tell me the number of sentences below that are about dogs:

நான் நாய்க்கு சில முட்டைகளைக் கொடுத்தேன்
நான் பறவைக்கு கொஞ்சம் சோளம் கொடுத்தேன்
புலி பொமரேனியனைப் பார்த்தது
- Requirements:
["Response is a number"]
-----
- Instruction:
Interpret the regression statistics below:
Regression Statistics 
Multiple R 0.983840293 
R Square 0.967941722 
Adjusted R Square 0.958782213 
Standard Error 229390.3205 
Observations 10 
 
ANOVA 
 df SS MS F Significance F 
Regression 2 1.11213E+13 5.56067E+12 105.6761684 5.89917E-06 
Residual 7 3.68339E+11 52619919134 
Total 9 1.14897E+13 
 
 Coefficients Standard Error t Stat P-value Lower 95% Upper 95% Lower 95.0% Upper 95.0%
Intercept 310055.0534 483971.7741 0.640646976 0.542147832 -834356.3408 1454466.448 -834356.3408 1454466.448
Employee Count -60461.95689 19894.68211 -3.039101432 0.018870807 -107505.4047 -13418.5091 -107505.4047 -13418.5091
Store Square Footage 91.18363601 17.59860205 5.181299955 0.001279055 49.56955481 132.7977172 49.56955481 132.7977172
- Requirements:
[]
-----
- Instruction:
i need working apps script to search a place name in a column, and store the lat and long on other columns
- Requirements:
["Response is Google Apps Script code", "Response must be working code that can be run"]
-----
- Instruction:
Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...
- Requirements:
["Response is a list of bullet points", "Response summarizes main ideas of Jeff Walker's Product Launch Formula", "Response would be relevant to a growth marketing agency"]
-----
- Instruction:
In Java, I want to replace string like "This is a new {{object}} at {{place}}" with a Map, {{object: "student", "point 3, 4"}}, and get a result "This is a new student at point 3, 4". How can I do?
- Requirements:
["Response is Java code", "Response must replace strings using the provided mapping "]
-----
- Instruction:
{instruction}
- Requirements:
"""
    return prompt


def construct_task_decomposition_prompt(instruction):
    prompt = f"""I am trying to analyze to what extent human instructions provided to language models can be effectively expressed as the composition of sub-instructions. If a task can be decomposed, then it can be written as a tree of sub-instructions; the output of each sub-instruction will either be used in a later sub-instruction or used for producing the final output. If the instruction cannot be effectively decomposed, return "NO DECOMPOSITION". Most instructions cannot be effectively decomposed!! Otherwise, represent your instruction decomposition as a linearized list of sub-instructions. Refer to previous sub-instructions that are used as inputs to later sub-instructions by their zero-indexed index in the list, prefixed with a # (e.g. "Create a dictionary containing the results of #0 and #1"). Your list should contain at least one element with a reference to a previous sub-instruction (i.e. one element containing '#'). Note that we are decomposing *instructions* and not *solutions*.

Examples
--------

- Instruction:
Write a Japanese poem about robots that is at least 10 lines long and contains internal rhyme.
- Task Decomposition:
NO DECOMPOSITION
-----
- Instruction:
Which group of sentences below contains more sentences about dogs?:

Group A
நான் நாய்க்கு சில முட்டைகளைக் கொடுத்தேன்
புலி பொமரேனியனைப் பார்த்தது

Group B
நான் பறவைக்கு கொஞ்சம் சோளம் கொடுத்தேன்
நான் பூனைக்கு சில முட்டைகளைக் கொடுத்தேன்
- Task Decomposition:
["Cluster sentences into 2 groups", "Classify whether each sentence in the first group from #0 is about dogs", "Classify whether each sentence in the first group from #1 is about dogs", "Count the number of positive values from #1", "Count the number of positive values from #2", "Compare the integer values in #3 and #4"]
-----
- Instruction:
Interpret the regression statistics below:
Regression Statistics 
Multiple R 0.983840293 
R Square 0.967941722 
Adjusted R Square 0.958782213 
Standard Error 229390.3205 
Observations 10 
 
ANOVA 
 df SS MS F Significance F 
Regression 2 1.11213E+13 5.56067E+12 105.6761684 5.89917E-06 
Residual 7 3.68339E+11 52619919134 
Total 9 1.14897E+13 
 
 Coefficients Standard Error t Stat P-value Lower 95% Upper 95% Lower 95.0% Upper 95.0%
Intercept 310055.0534 483971.7741 0.640646976 0.542147832 -834356.3408 1454466.448 -834356.3408 1454466.448
Employee Count -60461.95689 19894.68211 -3.039101432 0.018870807 -107505.4047 -13418.5091 -107505.4047 -13418.5091
Store Square Footage 91.18363601 17.59860205 5.181299955 0.001279055 49.56955481 132.7977172 49.56955481 132.7977172
- Task Decomposition:
NO DECOMPOSITION
-----
- Instruction:
Summarize the 3 main ideas of Ayn Rand into a list of bullet points, returned as a JSON string
- Task Decomposition:
["Construct a list of bullet points as a summary", "Return the list from #1 formatted as a JSON string]
-----
- Instruction:
Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...
- Task Decomposition:
NO DECOMPOSITION
-----
- Instruction:
i need working apps script to search a place name in a column, and store the lat and long on other columns
- Task Decomposition:
NO DECOMPOSITION
-----
- Instruction:
In Java, I want to replace string like "This is a new {{object}} at {{place}}" with a Map, {{object: "student", "point 3, 4"}}, and get a result "This is a new student at point 3, 4
- Task Decomposition:
NO DECOMPOSITION
-----
- Instruction
how do I add multiple new columns in m for power query or power bi?
- Task Decomposition:
NO DECOMPOSITION
-----
- Instruction:
{instruction}
- Task Decomposition:
"""
    return prompt

# Written by GPT-4
def plot_list_lengths(arr, keyword):
    # Get the lengths of the sublists  
    lengths = [len(sublist) for sublist in arr]  
    # Count the frequencies of each length  
    frequency = Counter(lengths)  
    # Prepare the x and y values for the plot  
    x = list(range(0, 8))  # x values are 0 to 7  
    y = [frequency.get(i, 0) for i in x]  # y values are the frequencies, default to 0 if length does not exist  
    # Create the bar plot  
    fig, ax = plt.subplots()
    ax.bar(x, y)  
    ax.set_xlabel('List Size')  
    ax.set_xlabel('Frequency')  
    ax.set_title(f'Distribution of Numbers of {keyword}')  
    fig.savefig(f"/tmp/{keyword}_distribution.png")

def create_csv(instructions, tags_list, keyword, filename):
    num_tags = max([len(t) for t in tags_list])
    headers = ["Prompt"] + [f"{keyword} {i+1}" for i in range(num_tags)]
    csv_list = [headers]
    for i, sublist in enumerate(tags_list):
        sublist_str = [f'"{instructions[i]}"'] + [str(s) for s in sublist]
        if len(sublist_str) < num_tags:
            sublist_str.extend([""] * (num_tags - len(sublist_str)))
        csv_list.append(sublist_str)
    with open(filename, 'w', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)

def cluster_keywords(instructions, sentences, num_clusters = 60, embeddings_cache = None):
    if embeddings_cache is not None and os.path.exists(embeddings_cache):
        sentence_embeddings = pickle.load(open(embeddings_cache, 'rb'))
    else:
        lowercased_sentences = [s.lower() for s in sentences]
        encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        sentence_embeddings = encoder_model.encode(lowercased_sentences)
        pickle.dump(sentence_embeddings, open(embeddings_cache, 'wb'))

    best_clusters = None
    best_silhouette_score = -1
    for num_clusters in tqdm(range(2, 150)):
        clusterer = KMeans(init='k-means++', n_clusters=num_clusters, random_state=0)
        clusters = clusterer.fit(sentence_embeddings)
        sil_coeff = silhouette_score(sentence_embeddings, clusters.labels_, metric='euclidean')
        if sil_coeff > best_silhouette_score:
            best_silhouette_score = sil_coeff
            best_clusters = clusters

    cluster_groups = [[] for _ in set(best_clusters.labels_)]
    for i, cluster_idx in enumerate(best_clusters.labels_):
        cluster_groups[cluster_idx].append(sentences[i])
    return cluster_groups


if __name__ == "__main__":
    sharegpt = json.load(open("ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"))

    instructions = []
    skill_prompts = []
    requirement_prompts = []
    task_decomposition_prompts = []

    for conversation in tqdm(sharegpt):
        if len(conversation["conversations"]) == 0:
            continue
        if conversation["conversations"][0]["from"] != "human":
            continue
        instruction = conversation["conversations"][0]["value"]
        encoding_name = "cl100k_base"
        encoding = tiktoken.get_encoding(encoding_name)
        if len(encoding.encode(instruction)) > 1000:
            continue
        instructions.append(instruction)
        skill_prompts.append(construct_skills_prompt(instruction))
        requirement_prompts.append(construct_requirements_prompt(instruction))
        task_decomposition_prompts.append(construct_task_decomposition_prompt(instruction))
        if len(instructions) == 1000:
            break

    skills = query_gpt(skill_prompts, "skills")
    requirements = query_gpt(requirement_prompts, "requirements")
    task_decompositions = query_gpt(task_decomposition_prompts, "task_decompositions")
    json.dump(skills, open("/tmp/skills.json", "w"))
    json.dump(requirements, open("/tmp/requirements.json", "w"))
    json.dump(task_decompositions, open("/tmp/task_decompositions.json", "w"))


    instructions_for_decompositions = []
    decompositions_processed = []
    for i, t in enumerate(task_decompositions):
        if t == -99999:
            continue
        decompositions_list = t if t != 0 else ["no decomposition"]
        decompositions_processed.append(". ".join(decompositions_list))
        instructions_for_decompositions.append(instructions)

    instructions_for_requirements = []
    requirements_processed = []
    for i, rqts in enumerate(requirements):
        if rqts == -99999:
            continue
        for t in rqts:
            requirements_processed.append(t)
            instructions_for_requirements.append(instructions[i])

    instructions_for_skills = []
    skills_processed = []
    for i, skls in enumerate(skills):
        if skls == -99999:
            continue
        for r in skls:
            skills_processed.append(r)
            instructions_for_skills.append(instructions[i])

    plot_list_lengths(decompositions_processed, "Decompositions")
    plot_list_lengths(skills_processed, "Skills")
    plot_list_lengths(requirements_processed, "Requirements")

    create_csv(instructions_for_decompositions, decompositions_processed, "Decompositions", "/tmp/decompositions.csv")
    create_csv(instructions_for_skills, skills_processed, "Skills", "/tmp/skills.csv")
    create_csv(instructions_for_requirements, requirements_processed, "Requirements", "/tmp/requirements.csv")

    decomposition_clusters = cluster_keywords(instructions_for_decompositions, decompositions_processed, num_clusters=60, embeddings_cache = "/tmp/decomposition_embeddings.pkl")
    requirement_clusters = cluster_keywords(instructions_for_requirements, requirements_processed, num_clusters=60, embeddings_cache = "/tmp/requirement_embeddings.pkl")
    skill_clusters = cluster_keywords(instructions_for_skills, skills_processed, num_clusters=60, embeddings_cache = "/tmp/skill_embeddings.pkl")
    breakpoint()