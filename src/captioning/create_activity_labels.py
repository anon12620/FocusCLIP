import argparse
import os
import json
import numpy as np
import scipy.io as sio
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MPII annotations to JSON format for the trainval set of mmpose, including activity labels')
    parser.add_argument('--anno_mat', default='data/mpii/annotations/mpii_human_pose_v1_u12_1.mat',
                        type=str, help='path to original MPII annotations mat file')
    parser.add_argument('--anno_mmpose_dir', default='data/mpii/annotations/mmpose',
                        type=str, help='path to mmpose annotations json file (trainval set)')
    parser.add_argument('--save_path', default='data/mpii/annotations/ours/mpii_trainval_with_act.json',
                        type=str, help='path to save the new annotations json file')
    args = parser.parse_args()
    return args


def main(args):
    # Load the original annotations provided by MPII
    mpii = sio.loadmat(args.anno_mat, struct_as_record=False)['RELEASE'][0,0]
    annolist = mpii.__dict__['annolist']
    activity = mpii.__dict__['act']
    num_images = annolist.shape[1]
    print('original annotations (all): {}'.format(num_images))  # includes test set as well

    # Load the trainval annotations provided by mmpose (does not include activity labels)
    mmpose_dir = args.anno_mmpose_dir  # this is the directory containing all JSON files for MPII dataset from mmpose
    mmpose_train_json = os.path.join(mmpose_dir, 'mpii_train.json')
    with open(mmpose_train_json, 'r') as f:
        mpii_train = json.load(f)
        train_imgs = set([sample['image'] for sample in mpii_train])
        print('mmpose train: {} images, {} people'.format(len(train_imgs), len(mpii_train)))

    mmpose_val_json = os.path.join(mmpose_dir, 'mpii_val.json')
    with open(mmpose_val_json, 'r') as f:
        mpii_val = json.load(f)
        val_imgs = set([sample['image'] for sample in mpii_val])
        print('mmpose val: {} images, {} people'.format(len(val_imgs), len(mpii_val)))

    mmpose_test_json = os.path.join(mmpose_dir, 'mpii_test.json')
    with open(mmpose_test_json, 'r') as f:
        mpii_test = json.load(f)
        test_imgs = set([sample['image'] for sample in mpii_test])
        print('mmpose test: {} images, {} people'.format(len(test_imgs), len(mpii_test)))

    # Extract the labels from original dataset that are missing in mmpose annotations (activity labels)
    print('extracting activity labels from original MPII annotations...')
    activity_labels = []
    skipped_samples = []  # images with missing annotations in the original dataset (test set + some trainval)
    for ix in range(0, annolist.shape[1]):
        anno, act = annolist[0, ix].__dict__, activity[ix, 0].__dict__
        name = anno['image'][0,0].__dict__['name'][0]

        try:
            # Extract the labels from original dataset that are missing in mmpose annotations
            activity_labels.append({
                'image': name,
                'frame_sec': anno['frame_sec'][0,0],  # frame number
                'vididx': anno['vididx'][0,0],  # video number
                'act_name': act['act_name'][0],  # activity name
                'cat_name': act['cat_name'][0],  # activity category
                'act_id': act['act_id'][0,0],  # activity label
                'istrain': name in train_imgs,
            })
        except:
            skipped_samples.append(name)
            continue  # annotations for test set are not available for MPII, so we skip them

    # Create a new trainval set by combining the original mmpose annotations with the missing labels
    print('combining the original mmpose annotations with the missing labels...')
    mpii_trainval_with_act = []
    for sample in tqdm(mpii_train + mpii_val):
        name = sample['image']
        if name in skipped_samples:
            continue

        anno = activity_labels[[a['image'] for a in activity_labels].index(name)]
        sample.update(anno)

        # Fix data types (for correct JSON serialization)
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = v.tolist()

            if isinstance(v, np.uint8):
                sample[k] = int(v)

            if isinstance(v, np.uint16):
                sample[k] = int(v)

        mpii_trainval_with_act.append(sample)

    labeled_imgs = set([sample['image'] for sample in mpii_trainval_with_act])
    print('labeled samples: {} images, {} people'.format(len(labeled_imgs), len(mpii_trainval_with_act)))
    print('skipped {} ({} val, {} train, {} test) images containing {} people because of missing annotations in the original dataset'.format(
        len(skipped_samples), len(val_imgs.intersection(skipped_samples)), len(train_imgs.intersection(skipped_samples)),
        len(test_imgs.intersection(skipped_samples)),
        len(skipped_samples) - len(val_imgs.intersection(skipped_samples)) - len(train_imgs.intersection(skipped_samples))
    ))

    # Save the new trainval set
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)

    save_file = args.save_path
    if os.path.exists(save_file):
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_file = args.save_path.replace('.json', '_{}.json'.format(timestamp))

    print('saving new annotations to {}'.format(save_file))
    with open(save_file, 'w') as f:
        json.dump(mpii_trainval_with_act, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)