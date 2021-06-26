import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import torchaudio.transforms as transforms
import torch.nn.functional as F
import sklearn
from sklearn.model_selection import train_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create char_map & index_map
char_map = {}
index_map = {}
char_map[''] = 0
char_map[' '] = 1
index_map[1] = ' '
for i in range(26):
    char_map[chr(i+97)] = i+2
    index_map[i+2] = chr(i+97)

def text_to_int(text):
    output = []
    for c in text:
        if c in char_map.keys():
            ch = char_map[c]
        else:
            ch = char_map['']
        output.append(ch)
    return output

def int_to_text(labels):
    output = ""
    for i in labels:
        output += index_map[i] + " "
    return output

audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    if(data_type != "valid"):
        for (waveform,utterance) in data:
            if data_type == 'train':
                spec = audio_transforms(waveform).squeeze(0).transpose(0, 1)
            else:
                spec = audio_transforms(waveform).squeeze(0).transpose(0, 1)
            spectrograms.append(spec)
            label = torch.Tensor(text_to_int(utterance.lower()))
            labels.append(label)
            input_lengths.append(spec.shape[0]//2)
            label_lengths.append(len(label))

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms.to(device), labels.to(device), input_lengths, label_lengths
    else:
        for waveform in data:
            spec = audio_transforms(waveform).squeeze(0).transpose(0, 1)
            spectrograms.append(spec)
            input_lengths.append(spec.shape[0]//2)
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        return spectrograms.to(device),input_lengths



def Decoder(output, blank_label=28,collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(int_to_text(decode))
    return decodes

def store_decode_text(model,valid_dataset):
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=lambda x: data_processing(x, 'valid')
                    )
    model.eval()
    file = open("./decoded_text.txt","w")
    with torch.no_grad():
        for spec,input_len in valid_loader:
            output = model(spec.to(device)) # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            
            decode = Decoder(output.cpu())
            #print(decode)
            file.write(decode[0]+'\n')
        file.write('\n')
    file.close()

def make_dataset(dir_path,data_type='train'):
    print("making dataset:",data_type)
    file_names = os.listdir(dir_path)
    wave_list = []
    for fn in file_names:
        wave,sr = torchaudio.load(dir_path+"/"+fn)
        wave_list.append(wave)
        
    if(data_type=='train'):
        file = open("./train.csv")
        utterance_list = []
        line = file.readline()
        while line:
            line = file.readline()
            line = line.split(',')
            if(len(line) < 3):
                break
            utterance_list.append(line[2].split('\n')[0])
        zip_data = list(zip(wave_list,utterance_list))
        train_dataset,test_dataset = train_test_split(zip_data, random_state=21, train_size=0.8, shuffle=True)
        return train_dataset,test_dataset
    
    else:
        return wave_list