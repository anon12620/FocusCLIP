import json
import os
from collections import Counter

import language_tool_python as lt
import numpy as np
import spacy
import torch
from lexical_diversity import lex_div as ld
from nltk import bigrams
from nltk.tokenize import word_tokenize, sent_tokenize
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_kincaid_grade
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm

# Make sure to download this model beforehand
nlp = spacy.load('en_core_web_md')


def semantic_coherence(captions):
    coherence = []

    for caption in captions:
        doc = nlp(caption)
        sentences = list(doc.sents)
        similarities = []

        for i in range(len(sentences) - 1):
            similarity = sentences[i].similarity(sentences[i + 1])
            similarities.append(similarity)

        coherence.append(np.mean(similarities))

    return np.mean(coherence)


def grammatical_errors(captions):
    errors = []
    tool = lt.LanguageTool('en-US')
    for caption in captions:
        matches = tool.check(caption)
        errors.append(len(matches))
    return np.mean(errors) if len(errors) > 0 else 0


def repetition_score_ngram(text, n=2):
    words = word_tokenize(text.lower())  # Convert to lower case and tokenize
    n_grams = list(bigrams(words))
    n_gram_freq = Counter(n_grams)

    # Count the n-grams that appear more than once
    repetitive_n_grams = [freq for n_gram,
                          freq in n_gram_freq.items() if freq > 1]

    # Calculate the repetition score
    score = sum(repetitive_n_grams) / len(n_grams) if len(n_grams) > 0 else 0

    return score


def chunk_text(text, max_length):
    sentences = sent_tokenize(text)
    chunks = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        if len(words) <= max_length:
            chunks.append(sentence)
            continue

        chunk = []
        for word in words:
            chunk.append(word)
            if len(chunk) >= max_length:
                chunks.append(' '.join(chunk))
                chunk = []

    return chunks


def clip_score(data, images_dir='data/mpii/images', model='openai/clip-vit-large-patch14'):
    print('Calculating CLIP scores ...')
    scores = []
    metric = CLIPScore(model)
    metric = metric.to('cuda')

    for item in tqdm(data):
        # Load image and move to GPU
        image_path = os.path.join(images_dir, item['image'])
        image = Image.open(image_path).convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1)
        image = image.to('cuda')

        # Load caption and split into chunks if too long
        caption = item['description']
        caption_chunks = chunk_text(caption, 55)  # 77 is max length for CLIP

        # Calculate CLIP score for each chunk and average
        errors = 0
        caption_score = 0
        for chunk in caption_chunks:
            try:
                caption_score += metric(image, chunk).item()
            except KeyboardInterrupt:
                raise
            except:
                errors += 1
                continue

        if len(caption_chunks) == errors:
            continue

        if errors > 0:
            print(f'Error in {errors}/{len(caption_chunks)} chunks.')

        score = caption_score / len(caption_chunks)
        scores.append(score)

    return np.mean(scores)


def readability_score(captions):
    readability_scores = [flesch_kincaid_grade(c)
                          for c in captions]
    return np.mean(readability_scores)


def style_consistency(captions):
    """
    Calculate style consistency using the average cosine similarity of TF-IDF vectors.

    Parameters:
    - captions (list): List of caption strings.

    Returns:
    - float: Average cosine similarity of TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(captions)
    similarities = cosine_similarity(X)
    # Exclude self-similarities (diagonal of the matrix) by setting them to zero and taking the mean
    np.fill_diagonal(similarities, 0)
    avg_similarity = np.mean(similarities)
    return avg_similarity


def lexical_diversity(captions):
    """
    Calculate lexical diversity using MTLD.

    Parameters:
    - captions (list): List of caption strings.

    Returns:
    - float: Lexical diversity as measured by MTLD.
    """
    # Combine all captions into a single string
    all_text = " ".join(captions)

    # Tokenize the text into words
    tokens = all_text.split()

    # Calculate MTLD
    mtld_score = ld.mtld(tokens)

    return mtld_score


def repetition_score(captions):
    scores = [repetition_score_ngram(c, n=3) for c in captions]
    return np.mean(scores)


def caption_length_stats(sentences):
    lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    return min(lengths), max(lengths), np.mean(lengths), np.std(lengths)


def analyze_captions(files):
    report = {}
    for i, file in enumerate(files):
        model_name = file.split('/')[-2]
        with open(file, 'r') as f:
            data = json.load(f)
            captions = [d['description'] for d in data]
        print(f'Processing {len(captions)} {model_name} captions ...')

        print('1/8 Length statistics ...', end='\r')
        length_stats = caption_length_stats(captions)

        print('2/8 Readability ...', end='\r')
        readability = readability_score(captions)

        print('3/8 Style consistency ...', end='\r')
        style = style_consistency(captions)

        print('4/8 Lexical diversity ...', end='\r')
        lexical = lexical_diversity(captions)

        print('5/8 3-gram repetition ...', end='\r')
        repetition = repetition_score(captions)

        print('6/8 Semantic coherence ...', end='\r')
        coherence = semantic_coherence(captions)

        print('7/8 Grammatical errors ...', end='\r')
        errors = grammatical_errors(captions)

        # Calculate CLIP score
        print('8/8 CLIP score ...', end='\r')
        clip = clip_score(data)
        report[model_name] = {
            'Flesch-Kincaid Grade': f'{readability:.2f}',
            'TF-IDF Cosine Similarity': f'{style:.2f}',
            'Lexical Diversity (MTLD)': f'{lexical:.2f}',
            '3-gram Repetition': f'{repetition:.2f}',
            'Semantic Coherence': f'{coherence:.2f}',
            'Grammatical Errors': f'{errors:.2f}',
            'CLIP Score (MPII)': f'{clip:.2f}',
            'Length Range': f'[{length_stats[0]}-{length_stats[1]}]',
            'Length Mean': f'{length_stats[2]:.2f}',
            'Length Std': f'{length_stats[3]:.2f}',
        }
        print(model_name, report[model_name])

    return report


def main():
    data_dir = 'data/mpii_captions/'
    save_dir = 'data/mpii_captions/'
    splits = ['val', 'train']

    for split in splits:
        data_files = []
        for root, dirs, files in os.walk(data_dir):
            if f'{split}.json' in files:
                data_files.append(os.path.join(root, f'{split}.json'))

        print(f'Analyzing {split} data for {len(data_files)} models.')
        report = analyze_captions(data_files)

        print(f'Saving report for {split} data.')
        report_file = os.path.join(save_dir, f'{split}_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)


if __name__ == '__main__':
    main()
