import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def create_human_mask(image, keypoints_list):
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.float32)

    for keypoints in keypoints_list:
        visible_keypoints = [tuple(map(int, pt[:2]))
                             for pt in keypoints if pt[2] > 0]

        if len(visible_keypoints) < 2:
            continue

        min_x, min_y = np.min(visible_keypoints, axis=0)
        max_x, max_y = np.max(visible_keypoints, axis=0)

        center = ((min_x + max_x) // 2, (min_y + max_y) // 2)
        axes = ((max_x - min_x) // 2, (max_y - min_y) // 2)

        # Create Gaussian distribution
        y, x = np.indices((height, width))
        gaussian = np.exp(-((x - center[0])**2 / (2 * axes[0]
                          ** 2) + (y - center[1])**2 / (2 * axes[1]**2)))

        # Multiply ellipse mask and Gaussian distribution
        mask += gaussian

    mask = np.clip(mask, 0, 1)
    return mask


def create_body_part_mask(image, keypoints_list):
    # 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck,
    # 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.float32)

    for keypoints in keypoints_list:
        # Define body parts and indices of the keypoints
        body_parts = {
            'head': [7, 8, 9, 12, 13],
            'torso': [6, 7, 12, 13, 2, 3],
            'left_upper_arm': [13, 14, 7],
            'right_upper_arm': [12, 11, 7],
            'left_lower_arm': [14, 15, 13],
            'right_lower_arm': [11, 10, 12],
            'left_upper_leg': [3, 4, 6],
            'right_upper_leg': [2, 1, 6],
            'left_lower_leg': [4, 5, 3],
            'right_lower_leg': [1, 0, 2],
        }

        for part, indices in body_parts.items():
            part_keypoints = [keypoints[i]
                              for i in indices if keypoints[i][2] > 0]
            part_mask = create_human_mask(image, [part_keypoints])
            mask += part_mask

    mask = np.clip(mask, 0, 1)
    return mask


def main():
    dataset = json.load(
        open('data/mpii/annotations/mpii_trainval_captioned.json', 'r'))
    print("Loaded MPII dataset with {} samples.".format(len(dataset)))

    # sort by image name
    dataset = sorted(dataset, key=lambda x: x['image'])

    # Create the output directory for masks
    os.makedirs('data/mpii/body_part_masks', exist_ok=True)

    # Generate masks for each sample in the dataset
    for sample in tqdm(dataset):
        image_id = sample['image']
        image = cv2.imread(os.path.join('data/mpii/images', image_id))

        people = sample['people']
        keypoints = []
        for person in people:
            kpts = person['kpts']
            kpts_vis = person['kpts_vis']

            # Convert keypoints to the format expected by create_human_mask
            keypoints.append(
                np.array([kpts[i] + [kpts_vis[i]] for i in range(len(kpts))]))

        keypoints_list = np.array(keypoints)

        # Create the human mask
        mask = create_body_part_mask(np.array(image), keypoints_list)

        # Save the mask as an image
        idx = image_id.split('.')[0]
        mask_path = os.path.join('data/mpii/body_part_masks', f"{idx}.png")
        cv2.imwrite(mask_path, mask * 255)


if __name__ == '__main__':
    main()
