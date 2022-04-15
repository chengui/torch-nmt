
class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _dict = dict(*args, **kwargs)
        self.update(_dict)

    def update(self, dct):
        for key, val in dct.items():
            if hasattr(val, 'keys'):
                val = AttrDict(val)
            self[key] = val

    def set(self, key, value):
        return self.__setitem__(key, value)

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

class Config(AttrDict):
    def from_dict(self, dct):
        self.update(dct)

    def from_object(self, obj):
        self.from_dict(obj.__dict__)

    def from_pyfile(self, content):
        data = {}
        exec(content, data)
        self.from_dict(data)

    def from_json(self, content):
        import json
        data = json.loads(content)
        self.from_dict(data)

    def from_yaml(self, content):
        import yaml
        data = yaml.safe_load(content)
        self.from_dict(data)

    def from_ini(self, content):
        import configparser
        config = configparser.ConfigParser()
        config.read_string(content)
        data = dict(config._sections)
        for k in data:
            data[k] = dict(data[k])
        self.from_dict(data)

    @classmethod
    def load_config(cls, fpath):
        import os.path as path
        config = cls()
        with open(fpath, 'r') as rf:
            content = rf.read()
            _, ext = path.splitext(fpath)
            if ext == '.json':
                config.from_json(content)
            elif ext == '.yaml':
                config.from_yaml(content)
            elif ext == '.ini':
                config.from_ini(content)
            elif ext == '.py':
                config.from_pyfile(content)
            else:
                raise ValueError('Unsupported file type: {}'.format(ext))
        return config
