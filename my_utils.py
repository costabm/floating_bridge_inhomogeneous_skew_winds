import numpy as np

def normalize(arr, old_bounds, new_bounds):
    """Normalize a 1D array, from old_bounds [min, max] to new desired bounds [new_max, new_min]"""
    return new_bounds[0] + (arr - old_bounds[0]) * (new_bounds[1] - new_bounds[0]) / (old_bounds[1] - old_bounds[0])

