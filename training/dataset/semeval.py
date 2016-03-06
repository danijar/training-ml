import re
import numpy as np
from dataset.dataset import Dataset


class SemEval(Dataset):

    urls = ['http://semeval2.fbk.eu/semeval2.php?location=download&'
            'task_id=11&datatype=trial']

    def parse(self, file_):
        pass

    def retrieve_from_sem_eval(self, path):
        with open(path) as infile:
            instring = infile.read()
            examples = instring.split('\n\n')
            relations = [self.create_relation(example) for
                         example in examples]
            return relations

    def create_relation(self, example):
        sent, label, comment = example.split('\n')
        cleared_sent = self.preprocess_sentence(sent)
        first = cleared_sent.find('E1')
        second = cleared_sent.find('E2')
        return Relation(cleared_sent, first, second, label)

    def clear_line(self, line):
        line = re.sub(r"<e1>(.*)</e1>", "E1", line)
        line = re.sub(r"<e2>(.*)</e2>", "E2", line)
        return line


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


class Relation:
    def __init__(self, sent, first_pos, second_pos, label):
        self.words = sent.split()
        self.first_pos = first_pos
        self.second_pos = second_pos
        self.label = label

    def __str__(self):
        words = self.sent.split()
        first_entity = " ".join(
            words[self.first_pos[0]: self.first_pos[1] + 1])
        second_entity = " ".join(
            words[self.second_pos[0]: self.second_pos[1] + 1])
        relation_str = "A relation between {} and {}".format(
            first_entity, second_entity)
        return relation_str
        pass
