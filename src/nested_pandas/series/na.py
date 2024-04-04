"""Missing value for NestedDtype

It i something between pandas' NA and NaN
"""

__all__ = ["NAType", "NA"]


class _NAType:
    pass


class NAType:
    """Singleton class representing missing value for NestedDtype.

    It doesn't implement most of the arithmetics and boolean logic operations,
    because they are ambiguous for missing values.

    The implementation is inspired both by pandas' NA and float number NaN.

    `NA` is a singleton instance of this class.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance of NAType."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<NA>"

    def __format__(self, format_spec) -> str:
        try:
            return self.__repr__().__format__(format_spec)
        except ValueError:
            return self.__repr__()

    def __bool__(self):
        raise TypeError("boolean value of NA is ambiguous")

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __hash__(self):
        return 0


NA = NAType()
"""Missed value for NestedDtype, a singleton instance of `NAType` class."""
