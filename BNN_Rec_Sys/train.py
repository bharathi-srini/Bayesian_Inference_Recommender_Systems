import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

import create_data, feature_engineering, item2vec_embedding

print('imported files')

#Function to read in csv files of Instacart data and prepare dataset
prepare_data()