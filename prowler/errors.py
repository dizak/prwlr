# -*- coding: utf-8 -*-


class ParserError(Exception):
    """
    Inappropriate structure passed to parser.
    """
    pass


class ProfileLengthError(Exception):
    """
    Different length of the reference and query.
    """
    pass
