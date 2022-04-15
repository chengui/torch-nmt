import os

class SubDir(object):
    def __init__(self, root='.'):
        self.root = root

    def dir(self, d):
        d = os.path.join(self.root, d)
        if not os.path.exists(d):
            os.makedirs(d)
        return d

    def file(self, f):
        f = os.path.join(self.root, f)
        return f

    def rfile(self, f):
        f = os.path.join(self.root, f)
        if not os.path.exists(f):
            raise OSError(f'File Not Found: {f}')
        return f

class WorkDir(SubDir):
    def __init__(self, root='.'):
        super().__init__(root)

    @property
    def data(self):
        return SubDir(self.dir('data'))

    @property
    def vocab(self):
        return SubDir(self.dir('vocab'))

    @property
    def model(self):
        return SubDir(self.dir('model'))

    @property
    def test(self):
        return SubDir(self.dir('test'))

    @property
    def out(self):
        return SubDir(self.dir('out'))