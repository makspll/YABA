from enum import Enum


class StringableEnum(Enum):
    """ class which enables easy string to Enum, Enum value to string conversion """
    @classmethod
    def from_str(cls, label: str) -> "StringableEnum":
        return cls[label.strip().upper().replace(' ','_')]

    @classmethod
    def to_str_from_value(cls, value) -> str:
        return cls(value).name.lower().replace('_',' ')
