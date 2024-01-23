

import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from sklearn.preprocessing import LabelEncoder
import pickle


import cv2
from mtcnn.mtcnn import MTCNN

def extract_faces_mtcnn(image_path, output_directory):
    img = cv2.imread(image_path)

    # Detect faces in the image using MTCNN
    faces = mtcnn_detector.detect_faces(img)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Extract and save each face to the output directory
    for i, face_info in enumerate(faces):
        x, y, w, h = face_info['box']
        face_roi = img[y:y + h, x:x + w]
        face_filename = os.path.join(output_directory, f"face_{i + 1}_{os.path.basename(image_path)}")
        cv2.imwrite(face_filename, face_roi)
        print(f"Face {i + 1} extracted from {os.path.basename(image_path)} and saved to {face_filename}")

def extract_faces_from_directory(input_directory, output_directory):
    # Loop through all files and subdirectories in the given directory
    for root, dirs, files in os.walk(input_directory):
        for filename in files:
            # Check if the file is an image (you might need to adjust this condition)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                
                # Specify the output directory based on the relative path from the input directory
                relative_path = os.path.relpath(image_path, input_directory)
                output_subdirectory = os.path.join(output_directory, os.path.dirname(relative_path))

                # Call the function to extract faces using MTCNN
                extract_faces_mtcnn(image_path, output_subdirectory)

# Specify the input directory containing images and subdirectories
input_directory = './datasets'

# Specify the output directory for extracted faces
output_directory = './extractImg'

# Load the pre-trained MTCNN model
mtcnn_detector = MTCNN()

# Extract faces from all images in the input directory and its subdirectories
extract_faces_from_directory(input_directory, output_directory)
