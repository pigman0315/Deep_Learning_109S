import torch
import os
import torch.nn as nn
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import random
import time
import difflib
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
#
import sklearn
from sklearn.model_selection import train_test_split
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Currently using GPU:",torch.cuda.get_device_name(0))

label_list = ["calendar","play","qa","email","iot","general","weather","lists","audio","alarm"]

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, num_layers, hidden_size, num_class, embed_dim, dropout_rate):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        #self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc_emb = nn.Linear(embed_dim,embed_dim)
        #
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first = True, # use batch size as first dimension, x -> (batch_size, seq, input_size)
            dropout = dropout_rate,
        )
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, num_class),
            )
        
    def forward(self, text, offsets):
        out = self.embedding(text, offsets)
        out = out.view(-1,1,out.size(1))
        out, _ = self.lstm(out)
        out = out[:,-1,:]
        out = self.fc(out)
        return out

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

def collate_batch_test(batch):
    text_list, offsets = [], [0]
    for _text in batch:
        processed_text = torch.tensor(text_encode(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return text_list.to(device), offsets.to(device)

class TitleCategorization:
    def __init__(self,file_path):
      self.prepare_data(file_path)

    def prepare_data(self,path):
      train_texts, train_labels = self.get_train_data(path)
      train_tokenized_texts = self.get_token(train_texts)
      self.vocab = self.build_vocab(train_tokenized_texts)
      self.train_texts = train_tokenized_texts
      self.train_labels = train_labels

    def get_train_data(self,path):
      file = open(path)
      line = file.readline()
      labels = []
      texts = []
      while line:
        line = file.readline()
        line = line.split("\n")[0]
        line = line.split(",")
        if len(line) < 3:
          break
        labels.append(label_list.index(line[1]))
        texts.append(line[2])
      file.close()
      return texts,labels

    def get_token(self,texts):
      #tokenizer = get_tokenizer('basic_english')
      tokenizer = RegexpTokenizer(r'[a-z\-]+')
      stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
      tokenized_texts = []
      for text in texts:
        text = text.lower()
        tokens = tokenizer.tokenize(text)
        filtered_tokens = [w for w in tokens if not w in stop_words]
        tokenized_texts.append(filtered_tokens)
      return tokenized_texts

    def split_train_test(self,train_size=0.7):
      zip_data = list(zip(self.train_texts,self.train_labels))
      self.train_dataset,self.test_dataset = train_test_split(zip_data, random_state=21, train_size=train_size, shuffle=True)

    # build vocabulary
    def build_vocab(self,tokenized_texts):
      self.counter = Counter()
      for line in tokenized_texts:
          self.counter.update(line)
      vocab = Vocab(self.counter)

      return vocab

    def evaluate(self,dataloader):
        self.model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = self.model(text, offsets)
                loss = self.criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc/total_count

    def run(self,dataloader,epoch):
        self.model.train()
        total_acc, total_count = 0, 0
        log_interval = 100
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            predicted_label = self.model(text, offsets)
            loss = self.criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                  total_acc/total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

    def retrieve_tokens(self,line):
        tokenizer = RegexpTokenizer(r'[a-z\-]+')
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
        line = line.split("  ")
        text = ''
        for l in line:
          word = l.replace(" ","")
          text += word + " "
        tokens = tokenizer.tokenize(text)
        filtered_tokens = [w for w in tokens if not w in stop_words]
        return filtered_tokens

    def get_valid_data(self,path):
        valid_dataset = []
        file = open(path)
        while True:
          line = file.readline()
          if(line=='\n'):
            break
          tokens = self.retrieve_tokens(line)
          valid_dataset.append(tokens)
        return valid_dataset

    def get_test_result(self,path):
        # dataloader
        test_dataset = self.get_valid_data(path)
        test_dataloader = DataLoader(test_dataset, batch_size=1,shuffle=False, collate_fn=collate_batch_test)
        self.model.eval()
        total_acc, total_count = 0, 0
        
        result = []
        with torch.no_grad():
            for i, (text, offsets) in enumerate(test_dataloader):
                predicted_label = self.model(text, offsets)
                idx = predicted_label.argmax(1).item()
                result.append(idx)
                if(i%100 == 0):
                  print(i,'/',len(test_dataloader),'batches')

        # turn result from number back to string
        for i in range(len(result)):
            result[i] = label_list[result[i]]

        # write result to file
        file = open('./submission.csv','w')
        file.write('File,Category\n')
        for i in range(len(result)):
            file.write(str(i)+','+result[i]+'\n')
        file.close()

    def train(self,num_layers=1,num_class=10,embed_dim=128,dropout_rate=0.1,lr=2.5,batch_size=16,epochs=10,hidden_size=256):
        vocab_size = len(self.vocab)
        self.model = TextClassificationModel(vocab_size, num_layers, hidden_size, num_class, embed_dim, dropout_rate).to(device)
        
        # training setting
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)
        total_accu = None
        
        # dataloader
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size,
                                      shuffle=True, collate_fn=collate_batch)
        test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size,
                                      shuffle=False, collate_fn=collate_batch)
        # training
        for epoch in range(1, epochs + 1):
          epoch_start_time = time.time()
          self.run(train_dataloader,epoch)
          accu_val = self.evaluate(test_dataloader)
          if total_accu is not None and total_accu > accu_val:
              self.scheduler.step()
          else:
              total_accu = accu_val
          print('-' * 59)
          print('| end of epoch {:3d} | time: {:5.2f}s | '
                'valid accuracy {:8.3f} '.format(epoch,time.time() - epoch_start_time,accu_val))
          print('-' * 59)

def string_similar(s1,s2):
  return difflib.SequenceMatcher(None, s1, s2).quick_ratio()
def text_encode(x):
  encode = []
  for token in x:
    n = vocab[token]
    if(n >= 2):
      encode.append(n)
    else:
      max_score = 0.0
      max_k = ""
      for k in counter.keys():
        sim_score = string_similar(token,k)
        if(sim_score > max_score):
          max_score = sim_score
          max_k = k
      encode.append(vocab[max_k])
  return encode

# Hyperparameters
EPOCHS = 10 # epoch
LR = 2.5  # learning rate
BATCH_SIZE = 16 # batch size for training
NUM_LAYERS = 1
HIDDEN_SIZE = 256
DROP_OUT = 0.0
EMBED_DIM = 300
NUM_CLASS = 10

#
tc = TitleCategorization("./train.csv")
tc.split_train_test(0.7)

# text, label pipeline
vocab = tc.vocab
counter = tc.counter
text_pipeline = lambda x: [vocab[token] for token in x]
label_pipeline = lambda x: int(x)

# train
print("=== Training starts ===")
tc.train(NUM_LAYERS,NUM_CLASS,EMBED_DIM,DROP_OUT,LR,BATCH_SIZE,EPOCHS,HIDDEN_SIZE)

print("\n=== Converting text to label ===")
# get result
tc.get_test_result("./decoded_text.txt")

print("\n=== Text to Label Finished (check \'submission.csv\') ===")