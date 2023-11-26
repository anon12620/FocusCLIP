"""" This script samples 1000 images from the pose description dataset for human evaluation. """
import json
import os
import random

from nltk import sent_tokenize

# Set seed for reproducibility
random.seed(42)

# Define paths of data files
data_files = [
    'data/mpii/annotations/captioned/gpt-4/mpii_train.json',
    'data/mpii_captions/gpt-3.5-turbo-0301/train.json',
    'data/mpii_captions/gpt-3.5-turbo-0613/train.json',
    'data/mpii_captions/llama-2-70b-chat-hf/train.json',
]

llm_names = [
    'gpt-4',
    'gpt-3.5-turbo-legacy',
    'gpt-3.5-turbo',
    'llama-2',
]


# Load the data
data = []
for file in data_files:
    with open(file, 'r') as f:
        data.append(json.load(f))


# Sample 1000 images
images = [i['image'] for i in data[0]]
random.shuffle(images)
images = list(set(random.sample(images, 1000)))
print('Sampled 1000 images')


# Transform the data into key-value pairs of image-caption
# pairs, where the key is the image and the value is a list.
# This list contains tuples with the LLM name and a sentence
# from the caption.
image_captions = {}
for idx, d in enumerate(data):
    for i in d:
        image_id = i['image']
        if image_id in images:
            if image_id not in image_captions:
                image_captions[image_id] = []

            llm = llm_names[idx]
            caption = i['description']
            caption_sentences = sent_tokenize(caption)
            llm_sentence_pairs = [(llm, s, 0) for s in caption_sentences]
            random.shuffle(llm_sentence_pairs)

            image_captions[image_id].extend(llm_sentence_pairs)
            random.shuffle(image_captions[image_id])
print(f'Transformed data into image-caption pairs with {len(image_captions)} images')

# Write the data to a file
eval_dir = 'data/llm_eval/human/'
os.makedirs(eval_dir, exist_ok=True)
captions_file = os.path.join(eval_dir, 'captions.json')
with open(captions_file, 'w') as f:
    json.dump(image_captions, f, indent=4)

# Copy images to eval dir
image_dir = 'data/mpii/images'
copy_dir = os.path.join(eval_dir, 'images')
os.makedirs(copy_dir, exist_ok=True)
for image in image_captions.keys():
    os.system(f'cp {os.path.join(image_dir, image)} {eval_dir}')
