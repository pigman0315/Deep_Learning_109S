import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import librosa
#
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio.transforms as transforms
import torch.nn.functional as F
#
import sklearn
from sklearn.model_selection import train_test_split
#
from utils import TextTransform, data_processing, Decoder, store_decode_text, make_dataset
from model import  SpeechRecognitionModel
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Currently using GPU:",torch.cuda.get_device_name(0))
#
def train(model, device, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()
        if batch_idx % 100 == 0:
           print(batch_idx*BATCH_SIZE,"/",data_len,", training loss=",loss.item())


def test(model, device, test_loader, criterion, epoch):

    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for I, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

    print('Test loss:',test_loss,'\n')


def main(train_dataset,test_dataset,valid_dataset,pretrained_model=None,learning_rate=5e-4, batch_size=20, epochs=10, n_features=128):

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                batch_size=16,
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train')
                                )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                batch_size=16,
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'test')
                                )
    
    model = SpeechRecognitionModel(n_cnn_layers=3, n_rnn_layers=3, rnn_dim=512,n_class=29, n_feats=n_features).to(device)

    
    if(pretrained_model != None):
        model.load_state_dict(torch.load(pretrained_model))
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, 
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=epochs,
                                            anneal_strategy='linear')

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch)
        test(model, device, test_loader, criterion, epoch)
        torch.save(model.state_dict(), "./weight.pth")
    return model


N_FEATURES = 64 # DO NOT CHANGE THIS
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 50


if __name__ == '__main__':

    print("=== Reading data ===")
    train_dataset,test_dataset = make_dataset("./train",data_type="train")
    valid_dataset = make_dataset("./test",data_type="valid")

    print("\n=== Training starts ===")
    model = main(train_dataset=train_dataset,test_dataset=test_dataset,valid_dataset=valid_dataset,pretrained_model="./weight/weight_rnn512_cer0.2.pth",learning_rate=LR,batch_size=BATCH_SIZE,epochs=EPOCHS,n_features=N_FEATURES)
        
    print("\n=== Decoding test dataset ===")
    store_decode_text(model,valid_dataset)

    print("\n==================== Audio to Text Finished ====================")




