# -*- coding: utf-8 -*-


from __future__ import print_function

class ParserError(Exception):
    """
    Inappropriate structure passed to parser.
    """
    pass


class ProfileError(Exception):
    """
    Different length of the reference and query.
    """
    pass


class SelectionFailWarning(Warning):
    """
    Dataframe boolean selection failure.
    """
    pass


class ExperimentalFeature(Warning):
    """
    This is an experimental feature. It may not work as expected nor work at
    all.
    """
    pass
