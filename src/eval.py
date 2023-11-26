# Purpose: Evaluate FocusCLIP on the specified tasks.

# Usage:
# python src/eval.py

# Example output:
# Evaluating Action Recognition (Stanford40)...
# 100%|█████████████████████��
# Action Recognition (Stanford40) Acc@3: 0.00%
# ...

# Python std.
import argparse
import json
import os

# 3rd party
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Project files
from datasets import get_dataset
from models import build_from_cfg


def load_model_openai(model_name):
    # Load the pretrained model
    available_models = [
        'openai/clip-vit-base-patch16',
        'openai/clip-vit-base-patch32',
        'openai/clip-vit-large-patch14'
    ]
    assert model_name in available_models, \
        f"Model {model_name} is not supported. Available models: {available_models}"

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def load_model(model_cfg, model_weights):
    assert os.path.exists(model_weights), \
        "Path to pretrained weights must be specified for our own models."
    assert os.path.exists(model_cfg), \
        "Path to model config must be specified for our own models."

    from utils.config import cfg, update_config
    update_config(cfg, args=argparse.Namespace(cfg=model_cfg))
    model = build_from_cfg(cfg, lightning_weights=model_weights).cuda()

    processor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return model, processor


def predict_openai(model, processor, texts, image):
    inputs = processor(text=texts, images=image,
                       return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs


def predict(model, processor, text_features, image):
    image = processor(image).cuda().unsqueeze(0)
    image_features = model.encode_image(image)
    image_features = F.normalize(image_features, dim=1).detach().cpu()
    probs = (image_features @ text_features.T).softmax(dim=1)
    return probs


def get_text_features(model, texts, chunk_size=1):
    if len(texts) > chunk_size:
        text_features = []
        for i in tqdm(range(0, len(texts), chunk_size)):
            chunk_features = model.encode_text(
                texts[i:i + chunk_size]).detach().cpu()
            text_features.append(chunk_features)
        text_features = torch.cat(text_features, dim=0)
    else:
        text_features = model.encode_text(texts).detach().cpu()
    text_features = F.normalize(text_features, dim=1)
    return text_features


def eval_task(task, model_name="openai/clip-vit-base-patch32", model_weights=""):
    # Load the pretrained model
    if 'openai' in model_name:
        model, processor = load_model_openai(model_name)
    else:
        model, processor = load_model(model_name, model_weights)

    if task['name'] == 'Presence Detection (COCO-Limbs)':
        from tasks.zero_shot import LimbsDetection
        limb_task = LimbsDetection(
            cfg=None,
            weights=None,
            model=(model, processor),
            image_dataset=get_dataset('keypoints/coco', **{
                'root': 'data/human-body-keypoints/coco/2017',
                'year': '2017',
                'split': 'val',
                'transform': transforms.Compose([
                    transforms.ToTensor(),
                ]),
            }),
        )

        import pytorch_lightning as pl
        trainer = pl.Trainer(accelerator='gpu', devices=1)
        trainer.test(limb_task, verbose=False)
        return limb_task.metrics['accuracy_topk']

    # Get the task parameters
    print(f"Task: {task['name']}")
    dataset_name = task['dataset']['name']
    dataset_kwargs = task['dataset']['kwargs']
    get_class_id = task['dataset']['class_id']
    k = task['k']

    # Load the dataset
    print(f"Loading dataset {dataset_name}...")
    dataset = get_dataset(dataset_name, **dataset_kwargs)
    num_classes = task['dataset']['num_classes'](dataset)
    texts = task['dataset']['texts'](dataset)
    print(f"Number of classes: {num_classes}")

    if 'openai' not in model_name:
        text_features = get_text_features(model, texts)

    print("Starting evaluation...")
    predictions = []
    targets = []
    for image, anno in tqdm(dataset):
        if 'openai' in model_name:
            probs = predict_openai(model, processor, texts, image)
        else:
            probs = predict(model, processor, text_features, image)

        # Save the predictions and targets
        predictions.append(probs[0].tolist())
        targets.append(get_class_id(anno))

    # Compute the metric
    metric = Accuracy(task='multiclass', num_classes=num_classes, top_k=k)
    return metric(torch.tensor(predictions), torch.tensor(targets))


tasks = {
    # Action Recognition
    'action/kinetics400': {
        'name': 'Action Recognition (Kinetics400)',
        'dataset': {
            'name': 'kinetics',
            'kwargs': {
                'root': 'data/eval/kinetics400',
                'split': 'test'
            },
            'texts': lambda dataset: dataset.activity_captions,
            'class_id': lambda anno: anno['activity_id'],
            'num_classes': lambda dataset: dataset.num_activities
        },
        'k': 3,
    },
    'action/stanford40': {
        'name': 'Action Recognition (Stanford40)',
        'dataset': {
            'name': 'action/stanford40',
            'kwargs': {
                'root': 'data/eval/stanford_actions/raw',
                'split': 'test'
            },
            'texts': lambda dataset: dataset.activity_captions,
            'class_id': lambda anno: anno['activity_id'],
            'num_classes': lambda dataset: dataset.num_activities
        },
        'k': 3,
    },

    # Age Classification
    'age/emotic': {
        'name': 'Age Classification (Emotic)',
        'dataset': {
            'name': 'emotic',
            'kwargs': {
                'root': 'data/eval/emotic/raw',
                'split': 'train'
            },
            'texts': lambda dataset: dataset.captions_age,
            'class_id': lambda anno: anno['age_group_id'],
            'num_classes': lambda dataset: dataset.num_age_groups
        },
        'k': 1,
    },
    'age/fairface': {
        'name': 'Age Classification (FairFace)',
        'dataset': {
            'name': 'fairface',
            'kwargs': {
                'root': 'data/eval/fairface/raw',
                'split': 'val'
            },
            'texts': lambda dataset: dataset.captions_age,
            'class_id': lambda anno: anno['age_group_id'],
            'num_classes': lambda dataset: dataset.num_age_groups
        },
        'k': 1,
    },
    'age/lagenda': {
        'name': 'Age Classification (LAGENDA-Body)',
        'dataset': {
            'name': 'lagenda',
            'kwargs': {
                'root': 'data/eval/lagenda/raw',
                'split': 'person'
            },
            'texts': lambda dataset: dataset.captions_age,
            'class_id': lambda anno: anno['age_group_id'],
            'num_classes': lambda dataset: dataset.num_age_groups
        },
        'k': 1,
    },
    'age/lagenda-f': {
        'name': 'Age Classification (LAGENDA-Face)',
        'dataset': {
            'name': 'lagenda',
            'kwargs': {
                'root': 'data/eval/lagenda/raw',
                'split': 'face'
            },
            'texts': lambda dataset: dataset.captions_age,
            'class_id': lambda anno: anno['age_group_id'],
            'num_classes': lambda dataset: dataset.num_age_groups
        },
        'k': 1,
    },
    'age/utkface': {
        'name': 'Age Classification (UTKFace)',
        'dataset': {
            'name': 'utk',
            'kwargs': {
                'root': 'data/eval/utkface/raw',
                'split': 'val'
            },
            'texts': lambda dataset: dataset.captions_age,
            'class_id': lambda anno: anno['age_group_id'],
            'num_classes': lambda dataset: dataset.num_age_groups
        },
        'k': 1,
    },

    # Emotion Recognition
    'emotion/ferplus': {
        'name': 'Emotion Recognition (FER+)',
        'dataset': {
            'name': 'ferplus',
            'kwargs': {
                'root': 'data/eval/ferplus/raw',
                'split': 'test'
            },
            'texts': lambda dataset: dataset.captions_emotion,
            'class_id': lambda anno: anno['emotion_id'],
            'num_classes': lambda dataset: dataset.num_emotions
        },
        'k': 3,
    },
    'emotion/fer2013': {
        'name': 'Emotion Recognition (FER2013)',
        'dataset': {
            'name': 'fer',
            'kwargs': {
                'root': 'data/eval/fer2013/raw',
                'split': 'train'
            },
            'texts': lambda dataset: dataset.captions_emotion,
            'class_id': lambda anno: anno['emotion_id'],
            'num_classes': lambda dataset: dataset.num_emotions
        },
        'k': 3,
    },
    'emotion/emotic': {
        'name': 'Emotion Recognition (Emotic)',
        'dataset': {
            'name': 'emotic',
            'kwargs': {
                'root': 'data/eval/emotic/raw',
                'split': 'train'
            },
            'texts': lambda dataset: dataset.captions_emotion,
            'class_id': lambda anno: anno['emotion_id'],
            'num_classes': lambda dataset: dataset.num_emotions
        },
        'k': 3,
    },

    # Race Classification
    'race/utkface': {
        'name': 'Race Classification (UTKFace)',
        'dataset': {
            'name': 'utk',
            'kwargs': {
                'root': 'data/eval/utkface/raw',
                'split': 'val'
            },
            'texts': lambda dataset: dataset.captions_race,
            'class_id': lambda anno: anno['race_id'],
            'num_classes': lambda dataset: dataset.num_races
        },
        'k': 1,
    },
}


def eval_tasks(tasks_to_eval, model_name, model_weights, results_file):
    # Create results directory if it doesn't exist
    results_dir = os.path.dirname(results_file)
    os.makedirs(results_dir, exist_ok=True)

    # Load existing results if any
    results = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)

    # Get required tasks and evaluate
    required_tasks = {k: tasks[k] for k in tasks_to_eval if k in tasks}
    for task in required_tasks.values():
        task_name = task['name']
        task_key = f"{task_name} Acc@{task['k']}"

        # Skip if already evaluated
        if task_key in results:
            task_acc = results[task_key]
            print(f"{task_key}: {task_acc}")
            continue

        try:
            acc = eval_task(task, model_name, model_weights)
            acc = f"{(acc * 100):.2f}%"
        except Exception as e:
            print(f"Failed to evaluate {task_name}: {e}")
            continue

        # Append task results
        results[task_key] = acc

        # Save results
        with open(results_file, 'w') as f:
            results = dict(sorted(results.items(), key=lambda x: x[0]))
            json.dump(results, f, indent=4)

        print(f"{task_key}: {acc}")


def main(args):
    # Get the arguments
    task_names = args.tasks
    model_name = args.model
    model_weights = args.model_weights
    results_dir = args.results_dir

    if task_names == ['all'] or task_names == ['*']:
        task_names = list(tasks.keys())

    # Evaluate
    if model_weights == "":
        model_name_short = "CLIP-OA"
    else:
        model_name_short = os.path.splitext(os.path.basename(model_weights))[0]
    results_file = os.path.join(results_dir, model_name_short + '.json')
    eval_tasks(task_names, model_name, model_weights, results_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate FocusCLIP or OpenAI CLIP on the specified tasks.')
    parser.add_argument('--tasks', nargs='+', default=['all'],
                        help='The tasks to evaluate.')
    parser.add_argument('--model', default="openai/clip-vit-base-patch32",
                        help='The model to use.')
    parser.add_argument('--model_weights', default="", type=str,
                        help='The model weights to use. This is only used for our own models, \
                            in which case the --model argument should specify config file and \
                            path to pretrained weights should be specified here.')
    parser.add_argument('--results_dir', default="results",
                        help='The directory to save results.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
