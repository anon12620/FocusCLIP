import json

import numpy as np
import scipy.io as sio
from PIL import Image

emotic_annotations_mat = 'data/emotic/CVPR17_Annotations.mat'

mat = sio.loadmat(emotic_annotations_mat)
splits = ['train'] #, 'val', 'test']

for split in splits:
    split_annotations = []
    for annotation in mat[split][0]:
        a, b, c, d, e = annotation

        image_name = a[0]
        image_folder = b[0]
        image_dims = {'width': int(c[0][0][1]), 'height': int(c[0][0][0])}
        image_path = 'data/emotic/' + image_folder + '/' + image_name
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        if w == image_dims['height'] and h == image_dims['width']:
            image_dims = {'width': w, 'height': h}

        emotion_data = []
        for entry in e[0]:
            bbox = entry[0].tolist()[0]
            emotion_categories = [cat[0][0] for cat in entry[1][0][0][0]]
            assert len(emotion_categories) == 1
            emotion_categories = emotion_categories[0]
            valence = int(entry[2][0][0][0]) if not np.isnan(
                entry[2][0][0][0]) else -1
            arousal = int(entry[2][0][0][1]) if not np.isnan(
                entry[2][0][0][1]) else -1
            dominance = int(entry[2][0][0][2]) if not np.isnan(
                entry[2][0][0][2]) else -1
            gender = entry[3][0]
            age = entry[4][0]

            emotion_data.append({
                'bbox': bbox,
                'categories': emotion_categories,
                'valence': valence,
                'arousal': arousal,
                'dominance': dominance,
                'gender': gender,
                'age': age
            })

        split_annotations.append({
            'image_name': image_name,
            'image_folder': image_folder,
            'image_dims': image_dims,
            'emotion_data': emotion_data
        })

    # Save to JSON file
    with open(f'data/emotic/annotations_{split}.json', 'w') as f:
        json.dump(split_annotations, f, indent=4)
