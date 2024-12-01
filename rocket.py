import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sktime.datasets import load_from_tsfile
from sktime.transformations.panel.rocket import Rocket
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingWarmRestarts
import argparse
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
from mmt import MiniMegaTortoraDataset
import seaborn as sns
import pandas as pd
from rich import print as print_rich
from io import BytesIO
# from ssaUtils import get_summary_str,train_table,DiscreteWaveletTransform1,WaveletFeatureExtractor,save_evaluated_lc_plots,pad_to_size_interpolate,trainingProgressBar
from ssaUtils import trainingProgressBar
from torch.utils.data import TensorDataset,DataLoader
import argparse
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sktime.datasets import load_from_tsfile
from sktime.transformations.panel.rocket import Rocket
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2

from sklearn.preprocessing import LabelEncoder
import pywt
from scipy import interpolate

def ridge_classifier(X_train,X_test,y_train,y_test):
    rocket = Rocket()

    # Fit ROCKET on the training data and transform both train and test sets
    rocket.fit(X_train)
    print("rocket fitted")
    X_train_transform = rocket.transform(X_train)
    X_test_transform = rocket.transform(X_test)
    print("transformed ts data to enhance features")

    # Train a RidgeClassifierCV (recommended with ROCKET features)
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transform, y_train)

    # Predict on the test set and evaluate
    y_pred = classifier.predict(X_test_transform)

    # Classification report and accuracy
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return [classification_report(y_test,y_pred),accuracy_score(y_test,y_pred)]


def rocket_classifier(X_train,X_test,y_train,y_test):
    classifier = RocketClassifier(random_state=42)
    print("fitting")
    classifier.fit(X_train, y_train)

    print("predictions")
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return [classification_report(y_test,y_pred),accuracy_score(y_test,y_pred)]


def hivecote2_classifier(X_train,X_test,y_train,y_test):
    classifier = HIVECOTEV2()
    print("fitting")
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Evaluate the classifier
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return [classification_report(y_test,y_pred),accuracy_score(y_test,y_pred)]


def DiscreteWaveletTransform1(trackSeries):

    # tempSeries=list()
    transformedSeries = []
    for track in trackSeries:
        w=pywt.Wavelet('db4')
        cA,cD= pywt.dwt(track,w,'constant')
        combined = np.concatenate([cA, cD])  # Flattened vector
        transformedSeries.append(combined)
    return transformedSeries  


def pad_to_size_interpolate(array, target_size):
    """
    Pad or sample a 1D array to a target size using cubic interpolation.

    Parameters:
    -----------
    array : numpy.ndarray
        1D input array
    target_size : int
        Desired length of the output array
        
    Returns:
    --------
    numpy.ndarray
        Array padded or sampled to target_size using interpolation
    """
    array = np.asarray(array)
    if array.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")

    current_size = len(array)

    # If array is already the target size, return it
    if current_size == target_size:
        return array.copy()

    # Create new x coordinates for interpolation
    old_x = np.linspace(0, 1, current_size)
    new_x = np.linspace(0, 1, target_size)

    # Use cubic interpolation for smooth results
    f = interpolate.interp1d(old_x, array, kind='cubic', fill_value='extrapolate')

    # Sample the interpolated function at the new x coordinates
    return f(new_x)


def mmt_dataset_creator():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    satelliteNumber=[60,160,300]
    trackSize = 500      # Maximum sample points for each track
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    mmt=MiniMegaTortoraDataset(satNumber=satelliteNumber,periodic=True)

    x,y=mmt.load_data_new()
    classes=[[x] for x in np.unique(y)]
    print(np.unique(y,return_counts=True))
    print(classes)

    x=DiscreteWaveletTransform1(x)

    # x = DiscreteWaveletTransform(x, wavelet='db4', level=3)
    x=[pad_to_size_interpolate(array,trackSize) for array in x]


    #Numpy array conversion        
    x=np.array(x)
    y=np.array(y)

    cat=preprocessing.OneHotEncoder().fit(classes)
    y=cat.transform(y).toarray()

    # Train-Val-Test split
    x_train,x_test,y_train,y_test=train_test_split(x,y,
                                                shuffle=True,
                                                test_size=0.2,
                                                random_state=42,
                                                stratify=y)


    x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,
                                                shuffle=True,
                                                test_size=0.2,                                                                                                random_state=42,
                                                stratify=y_train)

    # Normalization
    scaler=StandardScaler()
    
    x_train=scaler.fit_transform(x_train)
    x_val=scaler.transform(x_val)
    x_test=scaler.transform(x_test)

    #Only use if you not use ConvLSTM
    x_train=np.expand_dims(x_train,axis=-1)
    x_val=np.expand_dims(x_val,axis=-1)
    x_test=np.expand_dims(x_test,axis=-1)

    return x_train,x_val,x_test,y_train,y_test,y_val


# Paths to the training and testing .ts files
train_file = '/home/atila/aliberk_ws/bitirme_ws/InsectSound/InsectSound_TRAIN.ts'
test_file = '/home/atila/aliberk_ws/bitirme_ws/InsectSound/InsectSound_TEST.ts'

# Load training and testing datasets
# X_train, y_train = load_from_tsfile(train_file)
# X_test, y_test = load_from_tsfile(test_file)
# print(f"Training Data Shape: {X_train.shape}, Training Labels Shape: {y_train.shape}")
# print(f"Testing Data Shape: {X_test.shape}, Testing Labels Shape: {y_test.shape}")

x_train,x_val,x_test,y_train,y_test,y_val = mmt_dataset_creator()
print("dataset created")

# x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.4)
# x_test_split, x_test, y_test_split, y_test = train_test_split(x_test, y_test, test_size=0.4)
print("start fitting")
rocket_results = rocket_classifier(x_train,x_test,y_train,y_test)

# hivecote2_results = hivecote2_classifier(x_train,x_test,y_train,y_test)