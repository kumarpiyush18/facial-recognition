from packages.face_recognition import FaceRecognition

import argparse

import sys
import os

if hasattr(sys, '_MEIPASS'):
    # Running as a PyInstaller bundled executable
    base_path = sys._MEIPASS
else:
    # Running as a script
    base_path = os.getcwd()

# print(base_path)

def main(directory, faces_embeddings_path, model_path, outputdir):
    # Your existing code goes here
    print(f"Directory: {directory}")
    print(f"Faces Embeddings Path: {faces_embeddings_path}")
    print(f"Model Path: {model_path}")
    print(f"Result Path: {outputdir}")
    app = FaceRecognition(directory=directory, faces_embeddings_path=faces_embeddings_path, model_path=model_path,
                          outputdir=outputdir)
    app.run()


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Description of your script")

    # Define command-line arguments with default values
    parser.add_argument("--directory", default=os.path.join(base_path,"sampleinput"), help="Path to the directory")
    parser.add_argument("--faces_embeddings_path", default=os.path.join(base_path,"assets","face_embeddings_extract.npz"),
                        help="Path to face embeddings")
    parser.add_argument("--model_path", default=os.path.join(base_path,"model","SVC.model_small_class.pkl"), help="Path to the model")
    parser.add_argument("--outputdir", default=os.path.join(base_path,"output"), help="Path to the output directory")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(directory=args.directory, faces_embeddings_path=args.faces_embeddings_path, model_path=args.model_path,
         outputdir=args.outputdir)
