from preprocess import get_dataset
from pydantic import BaseModel
import json
from dotenv import load_dotenv
import anthropic
from jinja2 import Template
from bs4 import BeautifulSoup
import re
import concurrent.futures
from tqdm import tqdm
from sklearn.metrics import classification_report
import pandas as pd


load_dotenv(".env")

class PromptInput(BaseModel):
    abstract: str
    keywords: str
    domain: str
    area: str
    fields_and_areas: str  = "" # will contain dump of all labels as json
    

train, test = get_dataset(clean_text=False)

estimated_test_cost = test['X'].apply(lambda x: len(x)/4).sum() / 1_000_000
estimated_train_cost = train['X'].apply(lambda x: len(x)/4).sum() / 1_000_000
print(f"Estimated cost of training: {estimated_train_cost}")
print(f"Estimated cost of testing: {estimated_test_cost}")

prompt_inputs = []
hierarchical_labels = {}
labels_for_prompt = {}
for _, row in test.iterrows():
    domain = row['Domain'].lower().strip()
    area = row['area'].lower().strip()
    prompt_inputs.append(PromptInput(abstract=row['X'], keywords=row['keywords'], domain=domain, area=area))
    if domain not in hierarchical_labels:
        hierarchical_labels[domain] = {}
    if domain not in labels_for_prompt:
        labels_for_prompt[domain] = []
    if area not in hierarchical_labels[domain]:
        hierarchical_labels[domain][area] = 1
    else:
        hierarchical_labels[domain][area] += 1
    if area not in labels_for_prompt[domain]:
        labels_for_prompt[domain].append(area)

# sort labels for prompt
for domain in labels_for_prompt:
    labels_for_prompt[domain] = sorted(labels_for_prompt[domain])
    

for prompt_input in prompt_inputs:
    prompt_input.fields_and_areas = json.dumps(labels_for_prompt, indent=2)


client = anthropic.Anthropic()

with open("prompt.jinja", "r") as f:
    prompt = Template(f.read())


def get_prediction(prompt_input):
    rendered_prompt = prompt.render(prompt_input)

    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": rendered_prompt
                    }
                ]
            }
        ]
    )

    area, field = None, None
    pattern = r'<(\w+)>(.*?)<\/\1>'

    matches = re.findall(pattern, message.content[0].text)

    for tag, content in matches:
        if tag == "area":
            area = content
        if tag == "field":
            field = content

    valid_prediction = False
    if field in labels_for_prompt:
        if area in labels_for_prompt[field]:
            valid_prediction = True

    return {
        "area": area,
        "field": field,
        "valid_prediction": valid_prediction,
        "gt_area": prompt_input.area,
        "gt_domain": prompt_input.domain
    }

def run_in_parallel(inputs):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for result in tqdm(executor.map(get_prediction, inputs), total=len(inputs)):
            results.append(result)
    return results

# Run the function
results = run_in_parallel(prompt_inputs)
results = pd.DataFrame(results)
results.to_pickle("llm_results.pkl")

print("Main field classification")
print(classification_report(results['gt_domain'], results['field']))

print("Overall results")
print(classification_report(results['gt_area'], results['area']))

import pdb; pdb.set_trace()
print()