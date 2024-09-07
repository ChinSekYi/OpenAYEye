"""
data_ingestion.py

This module handles the configuration and ingestion of training, testing, and raw data files.
It defines a DataIngestionConfig class for setting default paths and imports necessary components
for data transformation, model training, exception handling, logging, and printing bankruptcy 
outcomes.

Classes:
- DataIngestionConfig: Configuration class for data paths.
"""


import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import print_bankruptcy_outcome

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


@dataclass
class DataIngestionConfig:
    pass





