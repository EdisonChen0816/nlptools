# encoding=utf-8
import json


class Message(object):

    def __init__(self, text, sender_id=None, data=None):
        self.data = data or {}
        self.set("text", text)
        self.sender_id = sender_id

    def set(self, prop, info):
        self.data[prop] = info

    def get(self, prop, default=None):
        return self.data.get(prop, default)

    def append(self, prop, info):
        assert isinstance(info, list)
        data = self.data.get(prop, [])
        assert isinstance(data, list)
        data.extend(info)
        self.set(prop, data)

    @property
    def text(self):
        return self.data.get("text", "")

    def __repr__(self):
        return json.dumps(self.data, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    pass
