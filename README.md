# FocusCLIP: Multimodal Subject-Level Guidance for Zero-Shot Transfer in Human-Centric Tasks

## Abstract

We introduce FocusCLIP, which integrates subject-level guidance--a specialized mechanism for target-specific supervision--into the CLIP framework for enhanced zero-shot transfer on human-centric tasks. Our approach narrows the search space by leveraging auxiliary supervision, emphasizing subject-relevant image regions through Region-of-Interest (ROI) heatmaps that emulate human visual attention mechanisms. Combined with Large Language Models (LLMs) for context-rich pose descriptions, this leads to a more focused embedding alignment. In our experiments, FocusCLIP, trained with images from the MPII Human Pose dataset and enriched with our novel pose descriptions, surpassed a simple baseline by 4.78 points, achieving an average accuracy of 24.54\% across a variety of unseen datasets and human-centric tasks. When also provided with person heatmaps, the accuracy increased to 30.52\%, a further 5.98 point gain, underlining the efficacy of subject-focused supervision. Additionally, we release a high-quality MPII Pose Descriptions dataset, encouraging further research in the field. Our findings emphasize the potential of integrating subject-level guidance with general pretraining methods for enhanced performance in downstream tasks.

## Prerequisites

### Setting up the environment

Create a new virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

We used Python 3.10.12 on a machine with 1x NVIDIA TITAN Xp GPU and CUDA 12.2. The versions of individual packages are listed in `requirements.txt`.

### Downloading the datasets

We trained FocusCLIP on the [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de/) with additional heatmaps and pose descriptions we created. For evaluation, the following eight datasets were used:
- [EMOTIC](https://github.com/rkosti/emotic)
- [FairFace](https://github.com/joojs/fairface)
- [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- [FER+](https://github.com/microsoft/FERPlus)
- [Kinectics-400](https://github.com/cvdfoundation/kinetics-dataset)
- [LAGENDA](https://wildchlamydia.github.io/lagenda/)
- [Stanford 40 Actions](http://vision.stanford.edu/Datasets/40actions.html)
- [UTKFace](https://susanqq.github.io/UTKFace/)

The datasets can be downloaded from their respective websites. We provide data loaders in [`src/datasets`](src/datasets/) for all datasets. For datasets which provide annotations in MATLAB format, we provide conversion scripts in [`src/datasets/converters`](src/datasets/converters/) to convert them to JSON format used by our data loaders.

We also provide scripts for downloading some of the datasets in [`src/datasets/downloaders`](src/datasets/downloaders/) to simplify the process.

### Downloading the pretrained models

The FocusCLIP model trained on our MPII Pose Descriptions dataset will be made available on [this link](https://drive.google.com/drive/folders/1XeHlFi27OFhWgqiusGwxPFY4BigxSQNs?usp=sharing). The baseline CLIP model trained on the same dataset will also be provided. The downloaded `.ckpt` files should be placed in the `checkpoints` directory.

## Evaluation

To evaluate FocusCLIP on all datasets used in the paper, run the following command:

```bash
python src/eval.py \
  --cfg configs/focusclip.yaml \
  --weights checkpoints/FocusCLIP.ckpt \
  --results_file ./results/FocusCLIP.json
```

This will save the results in a JSON file, corresponding to the FocusCLIP column in Table 2 in the paper. The baseline CLIP can also be evaluated similarly. To evaluate only on specific tasks, please see [L149-160 in `src/eval.py`](src/eval.py#L149-L160). You can comment out the tasks you don't want to evaluate on.