#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Morfessor 2.0 - Python implementation of the Morfessor method
"""
import logging

__all__ = ['MorfessorException', 'ArgumentException', 'MorfessorIO', 'FlatcatIO',
           'MorfessorBaseline', 'FlatcatModel', 'MorfessorEMPrune',
           'main', 'flatcat_main', 'get_default_argparser', 'get_flatcat_argparser', 'main_evaluation',
           'get_evaluation_argparser', 'MorphUsageProperties', 'HeuristicPostprocessor']
µ

__version__ = '2.0.7'
__author__ = 'Sami Virpioja, Peter Smit, Stig-Arne Grönroos'
__author_email__ = "morpho@aalto.fi"

show_progress_bar = True

_logger = logging.getLogger(__name__)


def get_version(numeric=False):
    if numeric:
        return __version__
    return 'FlatCat {}'.format(__version__)


# The public api imports need to be at the end of the file,
# so that the package global names are available to the modules
# when they are imported.

# Morfessor Baseline and Morfessor EM+Prune
from .util.constructions.base import BaseConstructionMethods
from .util.constructions.parallel import ParallelConstructionMethods
from .util.exception import MorfessorException

from .cmd import *

# Morfessor FlatCat
from .loss.flatcat.encoding import FlatcatAnnotatedCorpusEncoding
from .models.flatcat import FlatcatModel
from .models.flatcat._common import AbstractSegmenter
from .models.flatcat.categorizationscheme import MorphUsageProperties, HeuristicPostprocessor, WORD_BOUNDARY, CategorizedMorph
from .models.flatcat.cmd import flatcat_main, get_flatcat_argparser
from .models.flatcat.io import FlatcatIO
from .models.flatcat.utils import _progress
