from typing import List, Tuple, Sequence
from dataclasses import dataclass

from .base import _ConstructionMethods, Iterator


@dataclass
class ParallelConstruction:
    graphemes: Sequence[str]  # Can be a list of strings, or a string (which is itself really a list of strings)
    phonemes: Sequence[str]


class ParallelConstructionMethods(_ConstructionMethods[ParallelConstruction]):

    def force_split_locations(self, construction: ParallelConstruction):
        return []

    def split_locations(self, construction: ParallelConstruction, start=None, stop=None) -> Iterator[Tuple[int,int]]:
        start = (0,0) if start is None else start
        end = (len(construction.graphemes), len(construction.phonemes)) if stop is None else stop

        for gi in range(start[0] + 1, end[0]):
            for pi in range(start[1] + 1, end[1]):
                yield gi, pi

    def split(self, construction: ParallelConstruction, loc: Tuple[int,int]) -> Tuple[ParallelConstruction, ParallelConstruction]:
        assert 0 < loc[0] < len(construction.graphemes)
        assert 0 < loc[1] < len(construction.phonemes)
        return (ParallelConstruction(construction.graphemes[:loc[0]], construction.phonemes[:loc[1]]),
                ParallelConstruction(construction.graphemes[loc[0]:], construction.phonemes[loc[1]:]))

    def splitn(self, construction: ParallelConstruction, locs) -> Iterator[ParallelConstruction]:
        if len(locs) > 0 and not hasattr(locs[0], '__iter__'):
            for p in self.split(construction, locs):
                yield p
            return

        prev = (0,0)
        for l in locs:
            assert prev[0] < l[0] < len(construction.graphemes)
            assert prev[1] < l[1] < len(construction.phonemes)
            yield ParallelConstruction(construction.graphemes[prev[0]:l[0]], construction.phonemes[prev[1]:l[1]])
            prev = l
        yield ParallelConstruction(construction.graphemes[prev[0]:], construction.phonemes[prev[1]:])

    def parts_to_splitlocs(self, parts: List[ParallelConstruction]) -> Iterator[Tuple[int,int]]:
        cur_len = [0, 0]
        for p in parts[:-1]:
            cur_len[0] += len(p.graphemes)
            cur_len[1] += len(p.phonemes)
            yield tuple(cur_len)

    def slice(self, construction: ParallelConstruction, start=None, stop=None) -> ParallelConstruction:
        start = (0,0) if start is None else start
        stop = (len(construction.graphemes), len(construction.phonemes)) if stop is None else stop
        return ParallelConstruction(construction.graphemes[start[0]:stop[0]], construction.phonemes[start[1]:stop[1]])

    def from_string(self, string) -> ParallelConstruction:
        g, p = string.split('/', 1)
        assert len(g) > 0
        assert len(p) > 0
        return ParallelConstruction(g, p)

    def to_string(self, construction: ParallelConstruction) -> str:
        return u"{}/{}".format(construction.graphemes, construction.phonemes)

    def corpus_key(self, construction: ParallelConstruction):
        return (construction.graphemes, construction.phonemes)

    def lex_key(self, construction: ParallelConstruction):
        a = []
        a.extend(construction.graphemes)
        a.extend(construction.phonemes)
        return tuple(a)

    def atoms(self, construction: ParallelConstruction):
        a = []
        a.extend(construction.graphemes)
        a.extend(construction.phonemes)
        return tuple(a)

    def is_atom(self, construction: ParallelConstruction) -> bool:
        pass
