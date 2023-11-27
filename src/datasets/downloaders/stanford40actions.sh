#!/bin/bash

# Script to download and prepare the Stanford 40 Actions dataset.

# Create a directory to hold the dataset.
mkdir -p data/stanford40
cd data/stanford40

# Download the dataset components.
# This includes the JPEG images, XML annotations, and image splits.
wget http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip
wget http://vision.stanford.edu/Datasets/Stanford40_XMLAnnotations.zip
wget http://vision.stanford.edu/Datasets/Stanford40_ImageSplits.zip

# Unzip the downloaded files.
# Stanford40_JPEGImages.zip contains the image files.
# Stanford40_XMLAnnotations.zip contains the corresponding XML annotations.
# Stanford40_ImageSplits.zip contains text files defining training and test splits.
unzip Stanford40_JPEGImages.zip
unzip Stanford40_XMLAnnotations.zip
unzip Stanford40_ImageSplits.zip

# Remove the zip files to save space as they are no longer needed.
rm Stanford40_JPEGImages.zip
rm Stanford40_XMLAnnotations.zip
rm Stanford40_ImageSplits.zip

echo "Stanford 40 Actions dataset successfully downloaded and prepared."
