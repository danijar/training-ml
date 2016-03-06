import numpy as np
from dataset.dataset import Dataset


class Embeddings:
    def __init__(self, embedding_path):
        self.embeddings = self._load_embeddings(embedding_path)

    def __getitem__(self, word):
        return np.array(self.embeddings[word])

    def _load_embeddings(self, embedding_path):
        embeddings = {}

        with open(embedding_path) as embedding_file:
            for line in embedding_file:
                splitted = line.split()
                embeddings[splitted[0]] = splitted[1:]
        return embeddings


class SemEval(Dataset):

    urls = ['http://...']

    def parse(self, file_):
        pass
