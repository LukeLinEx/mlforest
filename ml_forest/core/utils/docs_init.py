import pickle


local_path = "/Users/lukelin/Desktop/test_ml_forest/experiment/local_documents/database.pkl"

with open(local_path, 'rb') as fp:
    root_database = pickle.load(fp)
