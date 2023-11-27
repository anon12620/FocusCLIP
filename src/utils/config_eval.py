""" Configuration for evaluation tasks. """
from easydict import EasyDict as edict

__C = edict()
cfg_eval = __C

__C.TASKS = edict()

# Define task categories
__C.TASKS.ACTION = edict()  # Action recognition
__C.TASKS.AGE = edict()  # Age classification
__C.TASKS.EMOTION = edict()  # Emotion recognition
__C.TASKS.RACE = edict()  # Race classification

# Define action recognition tasks
__C.TASKS.ACTION.KINETICS400 = {
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
}
__C.TASKS.ACTION.STANFORD40 = {
    'name': 'Action Recognition (Stanford40)',
    'dataset': {
        'name': 'stanford40',
        'kwargs': {
            'root': 'data/eval/stanford_actions/raw',
            'split': 'test'
        },
        'texts': lambda dataset: dataset.activity_captions,
        'class_id': lambda anno: anno['activity_id'],
        'num_classes': lambda dataset: dataset.num_activities
    },
    'k': 3,
}

# Define age classification tasks
__C.TASKS.AGE.EMOTIC = {
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
}
__C.TASKS.AGE.FAIRFACE = {
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
}
__C.TASKS.AGE.LAGENDA = {
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
}
__C.TASKS.AGE.LAGENDA_F = {
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
}
__C.TASKS.AGE.UTKFACE = {
    'name': 'Age Classification (UTKFace)',
    'dataset': {
        'name': 'utkface',
        'kwargs': {
            'root': 'data/eval/utkface/raw',
            'split': 'val'
        },
        'texts': lambda dataset: dataset.captions_age,
        'class_id': lambda anno: anno['age_group_id'],
        'num_classes': lambda dataset: dataset.num_age_groups
    },
    'k': 1,
}

# Define emotion recognition tasks
__C.TASKS.EMOTION.EMOTIC = {
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
}
__C.TASKS.EMOTION.FER2013 = {
    'name': 'Emotion Recognition (FER2013)',
    'dataset': {
        'name': 'fer2013',
        'kwargs': {
            'root': 'data/eval/fer2013/raw',
            'split': 'train'
        },
        'texts': lambda dataset: dataset.captions_emotion,
        'class_id': lambda anno: anno['emotion_id'],
        'num_classes': lambda dataset: dataset.num_emotions
    },
    'k': 3,
}
__C.TASKS.EMOTION.FERPLUS = {
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
}

# Define the race classification tasks
__C.TASKS.RACE.FAIRFACE = {
    'name': 'Race Classification (FairFace)',
    'dataset': {
        'name': 'fairface',
        'kwargs': {
            'root': 'data/eval/fairface/raw',
            'split': 'val'
        },
        'texts': lambda dataset: dataset.captions_race,
        'class_id': lambda anno: anno['race_id'],
        'num_classes': lambda dataset: dataset.num_races
    },
    'k': 1,
}
__C.TASKS.RACE.UTKFACE = {
    'name': 'Race Classification (UTKFace)',
    'dataset': {
        'name': 'utkface',
        'kwargs': {
            'root': 'data/eval/utkface/raw',
            'split': 'val'
        },
        'texts': lambda dataset: dataset.captions_race,
        'class_id': lambda anno: anno['race_id'],
        'num_classes': lambda dataset: dataset.num_races
    },
    'k': 1,
}
