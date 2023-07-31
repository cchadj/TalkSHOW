'''
load config from json file
'''
import json
import os

import configparser


class Object():
    _config: dict

    def __init__(self, config:dict) -> None:
        self._config = config
        for key in list(config.keys()):
            if isinstance(config[key], dict):
                setattr(self, key, Object(config[key]))
            else:
                setattr(self, key, config[key])

    def as_dict(self):
        return self._config.copy()

def load_JsonConfig(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    
    return Object(config)


if __name__ == '__main__':
    config = load_JsonConfig('config/style_gestures.json')
    print(dir(config))