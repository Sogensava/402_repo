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


class CNNFeatureExtractor(nn.Module):
    
    def __init__(self, in_channels, base_filters=32, kernel_sizes=[5, 3, 3]):
        super(CNNFeatureExtractor, self).__init__()
        
        self.cnn_layers = nn.ModuleList()
        current_channels = in_channels
        
        for i, k_size in enumerate(kernel_sizes):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=current_channels,
                        out_channels=base_filters * (2**i),
                        kernel_size=k_size,
                        padding='same'
                    ),
                    nn.BatchNorm1d(base_filters * (2**i)),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            )
            current_channels = base_filters * (2**i)
            
        self.output_channels = current_channels
        
    def forward(self, x):
        # Expected input shape: (batch_size, sequence_length, channels)
        # Convert to (batch_size, channels, sequence_length) for Conv1d
        x = x.transpose(1, 2)
        
        for layer in self.cnn_layers:
            x = layer(x)
            
        # Convert back to (batch_size, sequence_length, channels)
        return x.transpose(1, 2)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) 
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )       
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            feature_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.feature_embedding = nn.Linear(feature_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
            for _ in range(num_layers)]
        )        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out = self.dropout(self.feature_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        feature_size,
        num_classes,
        embed_size=128,
        num_layers=3,
        forward_expansion=2,
        heads=4,
        dropout=0.1,
        device="cuda",
        max_length=1000,
        cnn_base_filters=32,
        cnn_kernel_sizes=[7,5, 3, 3]
    ):
        super(TimeSeriesTransformer, self).__init__()
        
        # Add CNN feature extractor
        self.feature_extractor = CNNFeatureExtractor(
            in_channels=feature_size,
            base_filters=cnn_base_filters,
            kernel_sizes=cnn_kernel_sizes
        )
        
        # Modify encoder input size to match CNN output channels
        self.encoder = Encoder(
            feature_size=self.feature_extractor.output_channels,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            device=device,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length
        )
        
        self.device = device
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, num_classes)
        )
        
    def make_mask(self, x):
        mask = torch.ones(x.shape[0], 1, 1, x.shape[1]).to(self.device)
        return mask
    
    def forward(self, x):
        # Extract features using CNN
        x = self.feature_extractor(x)
        
        # Apply transformer
        mask = self.make_mask(x)
        encoder_out = self.encoder(x, mask)
        
        # Classification
        out = self.classifier(encoder_out.mean(dim=1))
        return out

def train_epoch(model, dataloader, criterion, optimizer, device,progressBar):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        X, y = batch

        # y=y.type(torch.LongTensor)
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # _, predicted = torch.max(outputs.data, 1)
        _, predicted = torch.max(torch.sigmoid(outputs.data), 1)

        total += y.size(0)
        # correct += (predicted == y).sum().item()
        correct+=(predicted==torch.max(y.data,1).indices).sum().item()
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        progressBar.update(0,advance=1,train_acc=accuracy,train_loss=avg_loss)

    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device,progressBar):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            # y=y.type(torch.LongTensor)

            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            # _, predicted = torch.max(outputs.data, 1)
            _, predicted = torch.max(torch.sigmoid(outputs.data), 1)

            total += y.size(0)
            # correct += (predicted == y).sum().item()
            correct+=(predicted==torch.max(y.data,1).indices).sum().item()
            avg_loss = total_loss / len(dataloader)
            accuracy = 100 *correct / total
            progressBar.update(0,val_acc=accuracy,val_loss=avg_loss)

def dataset_to_np_array(dataset):
    size = dataset.shape
    print(size)
    new_array = np.zeros((size[0],600,1))
    for i in range(0,size[0]):
        for j in range(0,600):
            new_array[i][j] = dataset[i][0][j]
            new_array[i][j][0] = dataset[i][0][j]
    return new_array

train_file = '/home/atila/aliberk_ws/bitirme_ws/InsectSound/InsectSound_TRAIN.ts'
test_file = '/home/atila/aliberk_ws/bitirme_ws/InsectSound/InsectSound_TEST.ts'

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trackSize = 600     # Maximum sample points for each track
    EPOCHS = 200    # Number of epochs for training
    batchSize = 32      # batch size for training
    history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load training and testing datasets
    x_train, y_train = load_from_tsfile(train_file)
    x_test, y_test = load_from_tsfile(test_file)

    encoder = LabelEncoder()

    classes=[[x] for x in np.unique(y_test)]
    print(classes)
    print(y_train.shape)
    print(y_train[0])
        
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1,1)

    cat=preprocessing.OneHotEncoder().fit(classes)
    y_train=cat.transform(y_train).toarray()
    y_test =cat.transform(y_test).toarray()

    print(y_train.dtype)

    x_train = dataset_to_np_array(x_train.to_numpy())
    x_test = dataset_to_np_array(x_test.to_numpy())

    x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.2)
    # Ensure data shapes are correct
    print(f"Training Data Shape: {x_train.shape}, Training Labels Shape: {y_train.shape}")
    print(f"Testing Data Shape: {x_test.shape}, Testing Labels Shape: {y_test.shape}")

    print(y_train.dtype)
    
    tensor_train_x=torch.Tensor(x_train_split)
    tensor_train_y=torch.Tensor(y_train_split)

    tensor_val_x=torch.Tensor(x_val)
    tensor_val_y=torch.Tensor(y_val)

    tensor_test_x=torch.Tensor(x_test)
    # tensor_test_y=torch.Tensor(y_test)

    tensor_TrainDataset=TensorDataset(tensor_train_x,tensor_train_y)
    tensor_ValDataset=TensorDataset(tensor_val_x,tensor_val_y)

    train_dataloader=DataLoader(tensor_TrainDataset,batch_size=batchSize)
    val_dataloader=DataLoader(tensor_ValDataset,batch_size=batchSize)

    print("dataset manipulation done")
    
    model = TimeSeriesTransformer(
        feature_size=1,
        num_classes=10,
        device=device,
        embed_size=128,
        num_layers=3,
        forward_expansion=2,
        heads=8,
        cnn_base_filters=32,
        max_length=trackSize,
        cnn_kernel_sizes=[7,5, 3, 3],
        dropout=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    scheduler = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,  # Initial restart interval
    T_mult=2  # Multiply interval by 2 after each restart
    )
    # Training loop
    num_epochs = EPOCHS  # Increased from 10
    best_accuracy = 0
    best_train_acc=0
    start_time = datetime.datetime.now()

    for epoch in range(num_epochs):
        progressBar=trainingProgressBar(epoch+1,len(train_dataloader))
        with progressBar:
            train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device,progressBar)
            test_acc = evaluate(model, val_dataloader, criterion, device,progressBar)
            
            
            # if test_acc > best_accuracy:
            #     best_accuracy = test_acc
            #     torch.save(model.state_dict(), 'best_model.pth')
            # if train_acc > best_train_acc:
            #     best_train_acc = train_acc
            # history["train_loss"].append(train_loss)
            # history["train_acc"].append(train_acc)
            # history["val_acc"].append(test_acc)

    print(f"Best Train Acc: {best_train_acc:.4f}")
    print(f"Best Test Acc: {best_accuracy:.4f}")
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds")    
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    
    

    plt.show()