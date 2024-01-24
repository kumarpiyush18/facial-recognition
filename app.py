from packages.test import TestSample

app = TestSample(directory="./sampleinput",faces_embeddings_path="./assets/face_embeddings_extract.npz",model_path="./model/SVC.model_small_class.pkl")
app.run()
