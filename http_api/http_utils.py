import numpy as np
import pandas as pd
from flask import json


class DataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Series):
            return obj.values.tolist()
        if isinstance(obj, pd.DataFrame):
            return [row.to_dict() for _, row in obj.iterrows()]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def read_json(file_path):
    with open(file_path, "r") as json_file:
        return json.load(json_file)


def write_json(obj, file_path):
    with open(file_path, "w") as json_file:
        return json.dump(obj, json_file)
