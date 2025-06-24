from enum import Enum

class Shape(str, Enum):
    RECTANGLE = "Rectangle"
    ELLIPSE = "Ellipse"
    IMAGE = "Image"
    OBJECTS = "Objects"
    CROPPING = "Previous cropping"

class RemovalMethod(str, Enum):
    NO = "No"
    EDGES = "Edges"
    ALL = "All"
