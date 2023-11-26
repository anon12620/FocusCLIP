import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Convert JSON annotations so each sample represents one image instead of one person')
    parser.add_argument('--anno_json', default='data/mpii/annotations/ours/mpii_trainval_with_act.json',
                        type=str, help='path to JSON annotations file with activity labels')
    parser.add_argument('--save_path', default='data/mpii/annotations/ours/mpii_trainval_with_act_single.json',
                        type=str, help='path to save the new annotations json file')
    args = parser.parse_args()
    return args


def main(args):
    # Load the annotations
    with open(args.anno_json, 'r') as f:
        mpii_trainval_with_act = json.load(f)
        print('loaded {} images, {} people'.format(len(set([sample['image'] for sample in mpii_trainval_with_act])), len(mpii_trainval_with_act)))

    # Create a new annotation format where each image has a single annotation, with multiple people if necessary
    print('converting annotations format ...')
    mpii_trainval_with_act_single = {}
    for sample in mpii_trainval_with_act:
        name = sample['image']
        if name not in mpii_trainval_with_act_single:
            mpii_trainval_with_act_single[name] = {
                'image': name,
                'video_id': sample['vididx'],
                'video_frame': sample['frame_sec'],
                'activity_id': sample['act_id'],
                'activity': sample['cat_name'] + ', ' + sample['act_name'],
                'count': 0,
                'people': [],
                'istrain': sample['istrain'],
            }

        people_count = len(mpii_trainval_with_act_single[name]['people'])
        mpii_trainval_with_act_single[name]['people'].append({
            'id': people_count,
            'center': [float(p) for p in sample['center']],
            'scale': sample['scale'],
            'kpts': [[float(j) for j in joint] for joint in sample['joints']],
            'kpts_vis': [int(j) for j in sample['joints_vis']],
        })

        mpii_trainval_with_act_single[name]['count'] += 1

    # Convert to a list
    mpii_trainval_with_act_single = list(mpii_trainval_with_act_single.values())

    # Save the new annotations
    with open(args.save_path, 'w') as f:
        json.dump(mpii_trainval_with_act_single, f)
        print('saved to {}'.format(args.save_path))


if __name__ == '__main__':
    args = parse_args()
    main(args)