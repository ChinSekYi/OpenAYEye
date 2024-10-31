"""
utils.py

This module contains utility functions for handling object serialization,
model evaluation, and printing bankruptcy prediction outcomes.
"""

import json
import os
import pickle
import sys

import dill
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save an object to a file using pickle.

    Args:
        file_path (str): Path to the file where the object will be saved.
        obj (object): Object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path):
    """
    Load an object from a file using dill.

    Args:
        file_path (str): Path to the file containing the object.

    Returns:
        object: Loaded object.
    """
    try:
        # print(file_path)
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e


def read_column_mapping(file_path):
    with open(file_path, "r") as file:
        column_mapping = json.load(file)
    return column_mapping


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

print(list_files(os.getcwd()))  # Replace with your repository path
