# encoding=utf-8
import copy


def override_defaults(defaults, custom):
    cfg = copy.deepcopy(defaults) if defaults else {}
    if custom:
        cfg.update(custom)
    return cfg


class ComponentMetaclass(type):

    @property
    def name(cls):
        return cls.__name__


class Component(object, metaclass=ComponentMetaclass):
    defaults = {}

    def __init__(self, inputs, outputs, config=None):
        self.inputs = inputs
        self.outputs = outputs
        config = config or {}
        config["name"] = self.name
        self.config = override_defaults(self.defaults, config)

    @property
    def name(self):
        return type(self).name

    def fit(self, data_path, **kwargs):
        pass

    def evaluate(self, data):
        pass

    def process(self, message, *args, **kwargs):
        raise NotImplementedError()

    def can_process(self, message):
        return True

    def load(self, dirn):
        pass

    def save(self, dirn):
        pass


class ComponentBuilder(object):
    pass


if __name__ == "__main__":
    pass
