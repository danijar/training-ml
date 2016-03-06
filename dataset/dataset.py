import random
import os
import shutil
from urllib.requrest import urlopen


class Dataset:

    urls = []

    def __init__(self):
        self.data = Data()
        try:
            self.__dict__.update(pickle.load(self._filename()))
            return
        except IOError:
            filenames = [self._download(x) for x in type(self).urls]
            with [open(x) for x in filesnames] as files:
                self.data = self.parse(*files)
            pickle.dump(self.__dict__, self._filename())

    def parse(self, *files):
        """
        Parse the downloaded file objects passed as arguments. Return a Data
        object containing training and test data and targets.
        """
        raise NotImplementedError

    def random_batch(self, size):
        indices = random.sample(range(len(self.train_data)), size)
        data = np.array([self.train_data[x] for x in indices])
        target = np.array([self.train_target[x] for x in indices])
        return data, target

    @classmethod
    def _download(cls, url):
        _, filename = os.path.split(url)
        filename = os.path.join(cls._folder(), filename)
        print('Download', filename)
        with urlopen(url) as response, open(filename, 'wb') as file_:
            shutil.copyfileobj(response, file_)
        return filename

    @classmethod
    def _filename(cls):
        name = cls.__name__.lower()
        return os.path.join(self._folder(), name + '.pickle')

    @classmethod
    def _folder(cls, prefix='~/.dataset'):
        name = cls.__name__.lower()
        path = os.path.join(os.path.expanduser(prefix), name)
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                return
            raise
        return folder
