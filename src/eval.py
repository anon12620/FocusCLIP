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

# Project files
from utils.config_eval import cfg_eval
from datasets import get_dataset
from models import build_from_cfg


def load_model(model_cfg, model_weights):
    if not os.path.exists(model_cfg):
        raise FileNotFoundError('Config file not found.')
    if not os.path.exists(model_weights):
        raise FileNotFoundError('Model weights not found.')

    from utils.config import cfg, update_config
    update_config(cfg, args=argparse.Namespace(cfg=model_cfg))
    model = build_from_cfg(cfg, lightning_weights=model_weights).cuda()

    processor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return model, processor


def predict(model, processor, text_features, image):
    image = processor(image).cuda().unsqueeze(0)
    image_features = model.encode_image(image)
    image_features = F.normalize(image_features, dim=1).detach().cpu()
    probs = (image_features @ text_features.T).softmax(dim=1)
    return probs


def get_text_features(model, texts, chunk_size=1):
    if len(texts) > chunk_size:
        text_features = []
        for i in tqdm(range(0, len(texts), chunk_size), desc="Encoding text"):
            chunk_features = model.encode_text(
                texts[i:i + chunk_size]).detach().cpu()
            text_features.append(chunk_features)
        text_features = torch.cat(text_features, dim=0)
    else:
        text_features = model.encode_text(texts).detach().cpu()
    text_features = F.normalize(text_features, dim=1)
    return text_features


def eval_task(task, cfg, weights):
    # Load the pretrained model
    model, processor = load_model(cfg, weights)

    # Get the task parameters
    print(f"Task: {task['name']}")
    dataset_name = task['dataset']['name']
    dataset_kwargs = task['dataset']['kwargs']
    get_class_id = task['dataset']['class_id']
    k = task['k']

    # Load the dataset
    print(f"Dataset: {dataset_name}")
    dataset = get_dataset(dataset_name, **dataset_kwargs)
    num_classes = task['dataset']['num_classes'](dataset)
    texts = task['dataset']['texts'](dataset)
    print(f"Num. Classes: {num_classes}")

    text_features = get_text_features(model, texts)

    predictions = []
    targets = []
    for image, anno in tqdm(dataset, desc="Evaluating"):
        probs = predict(model, processor, text_features, image)

        # Save the predictions and targets
        predictions.append(probs[0].tolist())
        targets.append(get_class_id(anno))

    # Compute the metric
    metric = Accuracy(task='multiclass', num_classes=num_classes, top_k=k)
    return metric(torch.tensor(predictions), torch.tensor(targets))


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
    for task in tasks_to_eval:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate FocusCLIP.')
    parser.add_argument('--cfg', default="configs/focusclip.yaml", type=str,
                        help='The config file of the model to evaluate.')
    parser.add_argument('--weights', default="", type=str,
                        help='The paths to the model weights.')
    parser.add_argument('--results_file', default="./output/results.json", type=str,
                        help='The path to the results file. Must be a JSON file. If \
                        the file exists, the results will be appended to it.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tasks = [  # Comment out the tasks you don't want to evaluate
        cfg_eval.TASKS.ACTION.KINETICS40,
        cfg_eval.TASKS.ACTION.STANFORD40,
        cfg_eval.TASKS.AGE.EMOTIC,
        cfg_eval.TASKS.AGE.LAGENDA,
        cfg_eval.TASKS.AGE.LAGENDA_F,
        cfg_eval.TASKS.AGE.UTKFACE,
        cfg_eval.TASKS.EMOTION.EMOTIC,
        cfg_eval.TASKS.EMOTION.FER2013,
        cfg_eval.TASKS.EMOTION.FERPLUS,
        cfg_eval.TASKS.RACE.FAIRFACE,
    ]
    eval_tasks(tasks, args.cfg, args.weights, args.results_file)
