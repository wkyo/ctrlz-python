import json
from json import dumps
from typing import Any, Sequence, IO


class StreamArray(list):
    """
    Converts a generator into a list object that can be json serialisable
    while still retaining the iterative nature of a generator.

    IE. It converts it to a list without having to exhaust the generator
    and keep it's contents in memory.
    """

    def __init__(self, generator):
        self.generator = generator
        self._len = 1

    def __iter__(self):
        self._len = 0
        for item in self.generator:
            yield item
            self._len += 1

    def __len__(self):
        """
        Json parser looks for a this method to confirm whether or not it can
        be parsed
        """
        return self._len


def dump_it(it: Sequence[Any], fp: IO[str], **json_kwargs) -> None:
    """Serialize a large sequence (or generator) to a json file"""
    fp.write('[')
    prev_is_obj = False
    for obj in it:
        if prev_is_obj:
            fp.write(',')
        else:
            prev_is_obj = True
        fp.write(dumps(obj, **json_kwargs))
    fp.write(']')


def dump_it_by_hack(it: Sequence[Any], fp: IO[str], **json_kwargs) -> None:
    """(Hack Version) Serialize a large sequence (or generator) to a json file

    This is a hack implemention! It relies heavily on the internal implementation
    of the `json.dump` method.
    """
    stream_array = StreamArray(it)
    for chunk in json.JSONEncoder(**json_kwargs).iterencode(stream_array):
        fp.write(chunk)


class JsonSequenceWriter:
    """Serialize a large sequence (or generator) to a json file"""

    def __init__(self, filepath: str, encoding: str = 'utf-8', **json_kwargs) -> None:
        self.filepath = filepath
        self._json_kwargs = json_kwargs

        self._fp = open(filepath, 'wt', encoding=encoding)
        self._count = 0

    def write(self, obj):
        if self._count > 0:
            self._fp.write(',')
        else:
            self._fp.write('[')
        self._fp.write(dumps(obj, **self._json_kwargs))
        self._count += 1

    def close(self):
        self._fp.write(']')
        self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # Exceptions will not be handled
        return False
