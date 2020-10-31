import pickle


def dump_pickle(obj: object, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)
