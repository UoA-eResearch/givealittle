#!/usr/bin/env python3

import pandas as pd
import requests
from pprint import pprint
import json5 as json # This is a more forgiving JSON parser that can handle comments, single quotes, and trailing commas
import torch
from PIL import Image
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from tqdm.auto import tqdm
import time
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

df = pd.read_excel("givealittle_health.xlsx")
done = pd.read_excel("LLM_results.xlsx")
#done = pd.DataFrame()  # Empty dataframe for now
df = df[~df.uri.isin(done.uri)]  # Remove rows that are already done

def get_text(row):
  text = ""
  if not pd.isna(row["title"]):
    text += "Title: " + row["title"] + "\n"
  if not pd.isna(row["pitch"]):
    text += "Pitch: " + row["pitch"] + "\n"
  if not pd.isna(row["description"]): # Description includes use_of_funds
    text += "Description: " + row["description"] + "\n"
  if not pd.isna(row["updates"]):
    text += "Updates: " + row["updates"] + "\n"
  if not pd.isna(row["location"]):
    text += "Location: " + row["location"] + "\n"
  return text.strip()

df["text"] = df.apply(get_text, axis=1)

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
    ICD: the top level ICD chapter for the primary health condition. One of:
        Chapter A00-B99 - Certain infectious and parasitic diseases
        Chapter C00-D49 - Neoplasms
        Chapter D50-D89 - Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
        Chapter E00-E89 - Endocrine, nutritional and metabolic diseases
        Chapter F01-F99 - Mental, Behavioral and Neurodevelopmental disorders
        Chapter G00-G99 - Diseases of the nervous system
        Chapter H00-H59 - Diseases of the eye and adnexa
        Chapter H60-H95 - Diseases of the ear and mastoid process
        Chapter I00-I99 - Diseases of the circulatory system
        Chapter J00-J99 - Diseases of the respiratory system
        Chapter K00-K95 - Diseases of the digestive system
        Chapter L00-L99 - Diseases of the skin and subcutaneous tissue
        Chapter M00-M99 - Diseases of the musculoskeletal system and connective tissue
        Chapter N00-N99 - Diseases of the genitourinary system
        Chapter O00-O9A - Pregnancy, childbirth and the puerperium
        Chapter P00-P96 - Certain conditions originating in the perinatal period
        Chapter Q00-Q99 - Congenital malformations, deformations and chromosomal abnormalities
        Chapter R00-R99 - Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified
        Chapter S00-T88 - Injury, poisoning and certain other consequences of external causes
        Chapter V00-Y99 - External causes of morbidity
        Chapter Z00-Z99 - Factors influencing health status and contact with health services
        Chapter U00-U85 - Codes for special purposes
    name: the name of the person this campaign is for
    gender: the gender of the person this campaign is for, one of Male, Female or Other/unknown
    age: the age of the person this campaign is for
    age_group: the age group of the person this campaign is for, one of 0-14, 15-64, 65+ or indeterminate/unknown
    ethnicity: the ethnicity of the person this campaign is for. If not mentioned in the text, guess their ethnicity from the image.
    urgency: a number from 0-100, indicating how urgent the need is
    sentiment: a number from 0-100, indicating the sentiment of the text, where 100 is the most positive, and 0 is the most negative
    truth: a number from 0-100, indicating how truthful the text is, where 100 is the most truthful, and 0 is the least truthful
    notes: any additional information about how you processed this text, such as warnings or errors
    smiling: a boolean indicating whether the person in the image is smiling
    deservingness: a number from 0-100, indicating how deserving the person is of receiving funds, where 100 is the most deserving, and 0 is the least deserving
    attractiveness: a number from 0-100, indicating how attractive the person is, where 100 is the most attractive, and 0 is the least attractive
    use: The main use of the raised funds - one or more (comma separated) of: medical expenses, experimental therapies, travel expenses, lost wages
    region: The region in New Zealand where the person is located, one of: Northland, Auckland, Waikato, Bay of Plenty, Gisborne, Hawke's Bay, Taranaki, Manuwatū-Whanganui, Wellington, Tasman, Nelson, Marlborough, West Coast, Canterbury, Otago, Southland
    narrative_clarity: a number from 0-100, indicating how clear the narrative is, where 100 is the most clear, and 0 is the least clear
    narrative_quality: a number from 0-100, indicating how well written the narrative is
    emotional_tone: grateful | desperate | hopeful | neutral | etc
    image_type: selfie | portrait | symbolic | environment | group | other
    face_visible: true | false
    facial_expression: smiling | neutral | serious | emotional | not_detectable
    image_quality: high | medium | low
    progression: a number from 0-100, indicating how advanced the condition is, where 100 is the most advanced, and 0 is the least advanced
    treatment: a number from 0-100, indicating how much treatment the person has received, where 100 is the most treatment, and 0 is the least treatment
    treatment_effectiveness: a number from 0-100, indicating how effective the treatment has been, where 100 is the most effective, and 0 is the least effective
    treatment_side_effects: a number from 0-100, indicating how severe the side effects of the treatment have been
    site: If the campaign is for cancer, what is the primary cancer site?
    stage: If the campaign is for cancer, what stage is the cancer at?
    reason: Summarise how donated funds will be used and the reason for requesting donations

    Do not include comments in your JSON response. Only respond with the JSON object. Make sure the JSON is valid
"""

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
     "Qwen/Qwen3-VL-30B-A3B-Instruct",
     dtype=torch.bfloat16,
     attn_implementation="flash_attention_2",
     device_map="auto",
     quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

results = []
# This will take a long time to run, so be patient. It processes 11,213 rows in about 62 hours.
# 100%|██████████| 11213/11213 [61:54:27<00:00, 19.88s/it]
for row in tqdm(df.itertuples(), total=len(df)):
    #print("Input:")
    #print(row.uri)
    #print(row.text)
    try:
        filepath = "hero_images/" + os.path.basename(row.hero) + ".jpg"
        image = Image.open(filepath)
    except Exception as e:
        print(e)
        continue
    #display(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": filepath},
                {"type": "text", "text": row.text}
            ]
        }
    ]
    try:
      inputs = processor.apply_chat_template(
          messages,
          tokenize=True,
          add_generation_prompt=True,
          return_dict=True,
          return_tensors="pt"
      )
      inputs = inputs.to(model.device)
    except Exception as e:
        print(f"Error processing row {row.Index}: {e}")
        continue

    for retry in range(3):
        try:
            generated_ids = model.generate(**inputs, max_new_tokens=5000)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            output_text = output_text.replace("```json", "").replace("```", "").strip()
            #print("Output:")
            result = json.loads(output_text)
            row_dict = row._asdict()
            row_dict.update(result)
            #pprint(result)
            #print("\n")
            results.append(row_dict)
            break
        except Exception as e:
          try:
            result = eval(output_text)
            row_dict = row._asdict()
            row_dict.update(result)
            #pprint(result)
            #print("\n")
            results.append(row_dict)
            break
          except Exception as e2:
            print(f"Unable to parse: {result}")

    if row.Index % 100 == 0:
        print(f"Processed {row.Index} rows, saving results...")
        pd.concat([done, pd.DataFrame(results)]).to_excel("LLM_results.xlsx", index=False)

pd.concat([done, pd.DataFrame(results)]).to_excel("LLM_results.xlsx", index=False)
print("Done")
