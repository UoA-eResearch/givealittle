#!/usr/bin/env python3

import pandas as pd
import requests
from pprint import pprint
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm.auto import tqdm
import time
import os

df = pd.read_excel("givealittle_health.xlsx")
df.fillna("", inplace=True)
df["text"] = df.title.str.cat(df[["title", "pitch", "description", "use_of_funds", "updates"]].astype(str), sep=" ")

# It's best to cache images locally, to avoid running into rate limits. The hero_images folder is only 180MB in total for 11,205 images.
os.makedirs("hero_images", exist_ok=True)
for uri in tqdm(df.hero):
  try:
    filepath = "hero_images/" + os.path.basename(uri) + ".jpg"
    if not os.path.isfile(filepath):
      r = requests.get(uri)
      r.raise_for_status()
      with open(filepath, "wb") as f:
        f.write(r.content)
    image = Image.open(filepath)
  except Exception as e:
    print(e)

prompt = """
    The below message is text extracted from givealittle, a crowdfunding platform. It's a health related campaign.
    I've also included the hero image for the campaign.
    For the text below, extract the following information, in JSON format:
    condition: the primary health condition mentioned in the text
    ICD10: the ICD10 code for the primary health condition
    ICD: the top level ICD chapter for the primary health condition
    name: the name of the person this campaign is for
    gender: the gender of the person this campaign is for
    age: the age of the person this campaign is for
    age_group: the age group of the person this campaign is for, one of 0-14, 15-64, 65+
    ethnicity: the ethnicity of the person this campaign is for. If not mentioned in the text, guess their ethnicity from the image.
    urgency: a number from 0-100, indicating how urgent the need is
    sentiment: a number from 0-100, indicating the sentiment of the text, where 100 is the most positive, and 0 is the most negative
    truth: a number from 0-100, indicating how truthful the text is, where 100 is the most truthful, and 0 is the least truthful
    notes: any additional information about how you processed this text, such as warnings or errors
    smiling: a boolean indicating whether the person in the image is smiling
    deservingness: a number from 0-100, indicating how deserving the person is of receiving funds, where 100 is the most deserving, and 0 is the least deserving
    attractiveness: a number from 0-100, indicating how attractive the person is, where 100 is the most attractive, and 0 is the least attractive
    use: The primary use of the raised funds - one of: medical expenses, experimental therapies, travel expenses, lost wages

    Do not include comments in your JSON response. Only respond with the JSON object. Make sure the JSON is valid
"""

indices = []
messages = []
for row in tqdm(df.itertuples(), total=len(df)):
    # Check the image can be read before feeding it to the LLM
    try:
        filepath = "hero_images/" + os.path.basename(row.hero) + ".jpg"
        image = Image.open(filepath)
    except Exception as e:
        print(e)
        continue
    indices.append(row.Index)
    messages.append([{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": "file://" + filepath},
                {"type": "text", "text": row.text}
            ]
    }])


# Loading this model uses 64.2GB VRAM, so the model can be loaded on a single A100 80GB GPU.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
     "Qwen/Qwen2.5-VL-32B-Instruct",
     torch_dtype=torch.bfloat16,
     attn_implementation="flash_attention_2",
     device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", use_fast=True, padding_side="left")

results = []
chunksize = 5 # Lower this if you run out of VRAM
for chunk in tqdm(range(0, len(df), chunksize)):
    subset = messages[chunk:chunk+chunksize]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in subset
    ]
    image_inputs, video_inputs = process_vision_info(subset)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Batch Inference
    generated_ids = model.generate(**inputs, max_new_tokens=5000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    for output_text in output_texts:
        try:
            output_text = output_text.replace("```json", "").replace("```", "").strip()
            results.append(json.loads(output_text))
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            results.append({})

df_results = pd.DataFrame(results, index=indices)
print(df_results)
df = df.merge(df_results, left_index=True, right_index=True, suffixes=("", "_result"), how="inner")
print(df)
df.to_excel("LLM_results.xlsx", index=False)