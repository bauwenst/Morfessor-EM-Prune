from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Union, List, Iterable, TypeVar, Generic

import re


T = TypeVar("T")


class _ConstructionMethods(ABC, Generic[T]):
    @abstractmethod
    def force_split_locations(self, construction: T) -> Iterable[int]:
        pass

    @abstractmethod
    def split_locations(self, construction: T, start: int=None, stop: int=None) -> Iterator[int]:
        """
        Return all possible split-locations between start and end. Start and end will not be returned.
        """
        pass

    @abstractmethod
    def split(self, construction: T, loc: int) -> Tuple[str,str]:
        pass

    @abstractmethod
    def splitn(self, construction: T, locs) -> Iterator[str]:
        pass

    @abstractmethod
    def parts_to_splitlocs(self, parts: List[T]) -> Iterator[int]:
        pass

    @abstractmethod
    def slice(self, construction: T, start: Union[int,str]=None, stop: Union[int,str]=None):
        pass

    @abstractmethod
    def from_string(self, string: str):
        pass

    @abstractmethod
    def to_string(self, construction: T) -> str:
        pass

    @abstractmethod
    def corpus_key(self, construction: T) -> str:
        pass

    @abstractmethod
    def lex_key(self, construction: T) -> str:
        pass

    @abstractmethod
    def atoms(self, construction: T) -> Iterable[str]:
        pass

    @abstractmethod
    def is_atom(self, construction: T) -> bool:
        pass


class BaseConstructionMethods(_ConstructionMethods[str]):
    def __init__(self, force_splits=None, nosplit_re: str=None):
        """
        :param force_splits: force segmentations on the characters in the given list
        :param nosplit_re: regular expression string for preventing splitting in certain contexts
        """
        self._force_splits = set(force_splits) if force_splits is not None else set()
        self._nosplit = re.compile(nosplit_re, re.UNICODE) if nosplit_re is not None else None

    def force_split_locations(self, construction: str):
        prev = 0
        for i in range(len(construction)):
            if construction[i] in self._force_splits:
                if i-prev > 0:
                    yield i
                if i+1 < len(construction):
                    yield i+1
                prev = i+1

    def split_locations(self, construction: str, start: int=None, stop: int=None):
        """
        Return all possible split-locations between start and end. Start and end will not be returned.
        """
        start = start if start is not None else 0
        stop = stop if stop is not None else len(construction)
        start = start if start != 'start' else 0
        stop = stop if stop != 'stop' else len(construction)
        assert all(not isinstance(x, str) for x in (start, stop)), 'start "{}" stop "{}"'.format(start, stop)

        for i in range(start+1, stop):
            if self._nosplit and self._nosplit.match(construction[i-1:i+1]):
                continue
            yield i

    def split(self, construction: str, loc: int):
        assert 0 < loc < len(construction)
        return construction[:loc], construction[loc:]

    def splitn(self, construction: str, locs):
        if not hasattr(locs, '__iter__'):
            for p in self.split(construction, locs):
                yield p
            return

        prev = 0
        for l in locs:
            assert prev < l < len(construction)
            yield construction[prev:l]
            prev = l
        yield construction[prev:]

    def parts_to_splitlocs(self, parts):
        cur_len = 0
        for p in parts[:-1]:
            cur_len += len(p)
            yield cur_len

    def slice(self, construction, start=None, stop=None):
        start = start if start != 'start' else None
        stop = stop if stop != 'stop' else None
        #assert all(not isinstance(x, str) for x in (start, stop)), 'start "{}" stop "{}"'.format(start, stop)
        return construction[start:stop]

    def from_string(self, string):
        return string

    def to_string(self, construction):
        return construction

    def corpus_key(self, construction):
        return construction

    def lex_key(self, construction):
        return construction

    def atoms(self, construction):
        return construction

    def is_atom(self, construction):
        return len(self.corpus_key(construction)) == 1
