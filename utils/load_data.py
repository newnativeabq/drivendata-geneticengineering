# load_data.py


import os
import pandas as pd



class LoadSpec():
    def __init__(self, filepath, **kwargs):
        self.filepath = filepath
        if kwargs is not None:
            self.kwargs = kwargs
        else:
            self.kwargs = {}

    def __repr__(self):
        return f'<LoadSpec> {self.filepath} {self.kwargs}'



def make_path(*args):
    return os.path.join(*args)



def _load_spec(spec: LoadSpec):
    return pd.read_csv(spec.filepath, **spec.kwargs)


def _load_files(files):
    for key in files:
        files[key] = [_load_spec(spec) for spec in files[key]]


def load(data_path=None, files=None):
    
    assert (data_path is not None) or (files is not None), 'Must provide file loadspec or data_path'
    if files is None:
        files = {
            'all': [
                LoadSpec(make_path(data_path, name)) for name in os.listdir(data_path)
            ]
        }
    _load_files(files)
    return files