import argparse
import json
import os

from tqdm import tqdm


def generate_caption(sample, model_name, caption_length=512, 
                     excluded_keys=['video_id', 'video_frame', 'activity_id', 'image']):
    system_prompt = 'You are an expert human activity and pose analyzer with deep understanding of MPII Human Pose dataset, which has 16 keypoints in order: 0 - right ankle, 1 - right knee, 2 - right hip, 3 - left hip, 4 - left knee, 5 - left ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - right wrist, 11 - right elbow, 12 - right shoulder, 13 - left shoulder, 14 - left elbow, 15 - left wrist. Given a set of 2D keypoint coordinates from MPII dataset as (x,y) with -1 for invisible joints, you will precisely describe body poses in terms of relative limb locations. Your descriptions will follow this template: "There are [num2word($count)] people in image who are [getVerb($activity) parseName($activity)]. [General attributes describing $activity in keypoints context.]" For each person in image: "The [parseLocation($center,$scale)] person is [predictStateFromContext()] with their [limb]..." For each limb (left leg, right leg, left arm, right arm, torso, head): "[Describe how these limbs are positioned relative to other limbs, bend angles, and other similar pose information.]" Use concise, precise and gender-neutral language.'

    # Filter out the keys that are not needed for captioning
    sample_ = {k: v for k, v in sample.items() if k not in excluded_keys}

    # Create the prompt
    prompt = {k: v for k, v in sample_.items() if k != 'people'}
    people = []
    for p_i in sample_['people']:
        kpts = []
        for k, v in zip(p_i['kpts'], p_i['kpts_vis']):
            if v == 0:
                kpts.append([-1, -1])
            else:
                kpts.append([int(x) for x in k])

        p_i_dict = {k: v for k, v in p_i.items() if k not in [
            'kpts', 'kpts_vis']}
        p_i_dict['kpts'] = kpts
        people.append(p_i_dict)
    prompt['people'] = people
    prompt = json.dumps(prompt)

    # Generate the caption
    temperature = 0.5
    if model_name in ['gpt-3.5-turbo', 'gpt-4']:
        from ask_gpt import ask_gpt
        model_fn = ask_gpt
        model_kwargs = {
            'system_prompt': system_prompt,
            'user_prompt': prompt,
            'model': model_name,
            'temperature': temperature,
            'max_tokens': caption_length,
            'max_retries': 3,
        }
    else:
        from ask_hf import ask_hf
        model_fn = ask_hf
        model_kwargs = {
            'system_prompt': system_prompt,
            'user_prompt': prompt,
            'model': model_name,
            'temperature': temperature,
            'max_gen_len': caption_length,
        }

    caption = model_fn(**model_kwargs)

    # Add the caption to the sample
    sample_ = sample.copy()
    sample_['description'] = caption
    return sample_


def main(model_name, split='train', limit=None):
    # Assert that the split is valid
    assert split in ['train', 'val', 'trainval'], 'Invalid split'

    # Load split file
    split_file = f'data/mpii/splits/{split}.txt'
    with open(split_file, 'r') as f:
        split_images = f.read().splitlines()

    # Load the dataset
    print('Loading dataset')
    json_file = f'data/mpii/annotations/transformed/mpii_trainval_llm2.json'
    with open(json_file, 'r') as f:
        dataset = json.load(f)

    # Filter the dataset to only include the split images
    dataset = [sample for sample in dataset if sample['image'] in split_images]
    print(f'Loaded {len(dataset)} samples from {split} split')

    # Create the save file path
    save_file = f'data/mpii/annotations/captioned/{model_name}/mpii_{split}.json'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Load the existing captioned data if it exists
    captioned_data = []
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            captioned_data = json.load(f)

    # Remove invalid samples from the captioned data
    captioned_data = [sample for sample in captioned_data if sample['image'] in split_images]

    # Get uncaptioned data
    uncaptioned_data = [sample for sample in dataset if sample['image'] not in [s['image'] for s in captioned_data]]
    print('{}/{} samples have already been captioned'.format(len(captioned_data), len(dataset)))

    # Caption the remaining samples
    end = len(uncaptioned_data) if limit is None else min(len(uncaptioned_data), limit)
    print(f'Captioning next {end} samples')
    print(f'Save path: {save_file}')
    for sample in tqdm(uncaptioned_data[:end]):
        # Generate the caption
        captioned_sample = generate_caption(sample, model_name)
        captioned_data.append(captioned_sample)

        # Add the captioned sample to the save file
        with open(save_file, 'w') as f:
            json.dump(captioned_data, f)

    # Print the number of captioned samples
    print(f'Captioned {len(captioned_data)} samples')

    # Save the captioned data
    with open(save_file, 'w') as f:
        json.dump(captioned_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-2-70b-chat-hf',
                        help='Model to use for caption generation')
    parser.add_argument('--split', type=str, default='train',
                        help='Split to caption')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of samples to caption')
    args = parser.parse_args()

    main(args.model, args.split, args.limit)
