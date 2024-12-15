#!/usr/bin/env python3

import pandas as pd
import requests
from pprint import pprint
import json
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm.auto import tqdm

df = pd.read_excel("givealittle_health.xlsx")
df["text"] = df.title.str.cat(df[["pitch", "description", "use_of_funds"]].astype(str), sep=" ")

# Loading this model needs about 22.69GB of GPU memory
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

results = []
for row in tqdm(df.itertuples(), total=len(df)):
    #print("Input:")
    #print(row.uri)
    #print(row.text)
    try:
        image = Image.open(requests.get(row.hero, stream=True).raw)
    except Exception as e:
        print(e)
        continue
    #display(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": """
                    The below message is text extracted from givealittle, a crowdfunding platform. It's a health related campaign.
                    I've also included the hero image for the campaign.
                    For the text below, extract the following information, in JSON format:
                    condition: the primary health condition mentioned in the text
                    ICD10: the ICD10 code for the primary health condition
                    name: the name of the person this campaign is for
                    gender: the gender of the person this campaign is for
                    age: the age of the person this campaign is for
                    ethnicity: the ethnicity of the person this campaign is for. If not mentioned in the text, guess their ethnicity from the image.
                    urgency: a number from 0-100, indicating how urgent the need is
                    sentiment: a number from 0-100, indicating the sentiment of the text, where 100 is the most positive, and 0 is the most negative
                    truth: a number from 0-100, indicating how truthful the text is, where 100 is the most truthful, and 0 is the least truthful
                    notes: any additional information about how you processed this text, such as warnings or errors
                    smiling: a boolean indicating whether the person in the image is smiling

                    Do not include comments in your JSON response. Only respond with the JSON object. Make sure the JSON is valid.
                """},
                {"type": "image"},
                {"type": "text", "text": row.text}
            ]
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    for retry in range(3):
        output = model.generate(**inputs, max_new_tokens=5000)
        result = processor.decode(output[0])
        result = result[result.rindex("<|end_header_id|>") + len("<|end_header_id|>"):].strip().replace("<|eot_id|>", "")
        #print("Output:")
        try:
            result = json.loads(result)
            row = row._asdict()
            row.update(result)
            #pprint(result)
            #print("\n")
            results.append(row)
            break
        except json.JSONDecodeError:
            print(f"Unable to parse: {result}")

pd.DataFrame(results).to_excel("LLM_results.xlsx", index=False)