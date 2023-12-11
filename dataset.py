import csv
import re

import logging

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

label_m={'Expect':0, 'Hate':1, 'Surprise':2, 'Anger':3}
label_list = {'Love':0, 'Joy':1, 'Anxiety':2, 'Sorrow':3, 'Expect':4, 'Hate':5, 'Surprise':6, 'Anger':7}
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
num2label = {0:'Love', 1:'Joy', 2:'Anxiety', 3:'Sorrow', 4:'Expect', 5:'Hate', 6:'Surprise', 7:'Anger'}

class TextDataset(Dataset):
    def __init__(self,windows=False):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        if windows:
            csf=open("data/Train.csv","r",errors = 'ignore')
        else:
            csf = open("data/Train_linux.csv", "r", errors='ignore')
        csf_reader=csv.reader(csf)
        num=0
        maxlen=0
        self.examples=[]
        self.labels=[]
        self.texts=[]

        for i in csf_reader:
            if num == 0:
                num+=1
                continue
            num+=1
            id=int(i[0])
            text=i[1].replace(" ","")
            if len(text)>maxlen:
                maxlen=len(text)
            lable=i[2]
            onelable=[0,0,0,0,0,0,0,0]

            it = re.finditer(r"[a-zA-Z]+",lable)
            for match in it:
                onelable[label_list[match.group()]]=1
            self.labels.append(onelable)
            self.texts.append(text)
        print()
        print("maxlen:",maxlen)

    def __getitem__(self, idx):
        text=self.texts[idx]
        label=self.labels[idx]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(text,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=256,
                                      return_tensors='pt')  # Return torch.Tensor objects
        # shape [max_len]
        # tensor of token ids  torch.Size([max_len])
        token_ids = encoded_pair['input_ids'].squeeze(0)
        token_ids.cuda()
        # binary tensor with "0" for padded values and "1"  for the other values  torch.Size([max_len])
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        attn_masks.cuda()
        # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens torch.Size([max_len])
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        token_type_ids.cuda()
        label=torch.tensor(label)*1.0
        input_data={
            "token_ids":token_ids,
            "attn_masks":attn_masks,
            "token_type_ids":token_type_ids,
            "label":label
        }
        return input_data

    def __len__(self):
        return len(self.texts)
class TextDataset_last(Dataset):
    def __init__(self,windows=False):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        if windows:
            csf=open("data/Train.csv","r",errors = 'ignore')
        else:
            csf = open("data/Train_linux.csv", "r", errors='ignore')
        csf_reader=csv.reader(csf)
        num=0
        maxlen=0
        self.examples=[]
        self.labels=[]
        self.texts=[]

        for i in csf_reader:
            if num == 0:
                num+=1
                continue
            num+=1
            id=int(i[0])
            text=i[1].replace(" ","")
            if len(text)>maxlen:
                maxlen=len(text)
            lable=i[2]
            onelable=[0,0,0,0]

            it = re.finditer(r"[a-zA-Z]+",lable)
            for match in it:
                if match.group() in label_m:
                    onelable[label_m[match.group()]]=1
            self.labels.append(onelable)
            self.texts.append(text)
        print("maxlen:",maxlen)

    def __getitem__(self, idx):
        text=self.texts[idx]
        label=self.labels[idx]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(text,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=256,
                                      return_tensors='pt')  # Return torch.Tensor objects
        # shape [max_len]
        # tensor of token ids  torch.Size([max_len])
        token_ids = encoded_pair['input_ids'].squeeze(0)
        token_ids.cuda()
        # binary tensor with "0" for padded values and "1"  for the other values  torch.Size([max_len])
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        attn_masks.cuda()
        # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens torch.Size([max_len])
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        token_type_ids.cuda()
        label=torch.tensor(label)*1.0
        input_data={
            "token_ids":token_ids,
            "attn_masks":attn_masks,
            "token_type_ids":token_type_ids,
            "label":label
        }
        return input_data

    def __len__(self):
        return len(self.texts)




class test_TextDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        csf=open("data/test.csv","r",errors = 'ignore')

        csf_reader=csv.reader(csf)
        num=0
        maxlen=0
        self.examples=[]
        self.labels=[]
        self.texts=[]

        for i in csf_reader:
            if num == 0:
                num+=1
                continue
            num+=1
            text=i[1].replace(" ","")
            if len(text)>maxlen:
                maxlen=len(text)

            self.texts.append(text)
        print("maxlen:",maxlen)

    def __getitem__(self, idx):
        text=self.texts[idx]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(text,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=256,
                                      return_tensors='pt')  # Return torch.Tensor objects
        # shape [max_len]
        # tensor of token ids  torch.Size([max_len])
        token_ids = encoded_pair['input_ids'].squeeze(0)
        token_ids.cuda()
        # binary tensor with "0" for padded values and "1"  for the other values  torch.Size([max_len])
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        attn_masks.cuda()
        # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens torch.Size([max_len])
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        token_type_ids.cuda()
        input_data={
            "token_ids":token_ids,
            "attn_masks":attn_masks,
            "token_type_ids":token_type_ids,
            "text":text
        }
        return input_data

    def __len__(self):
        return len(self.texts)
    def write2csv(self,pred):
        with open("data/test_pred.csv","w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["id","label"])
            for i in range(len(pred)):
                writer.writerow([i+1,pred[i]])




class OneTextDataset(Dataset):
    def __init__(self,datalable):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        csf=open("data/Train.csv","r",errors = 'ignore')

        csf_reader=csv.reader(csf)
        num=0
        maxlen=0
        self.examples=[]
        self.labels=[]
        self.texts=[]

        for i in csf_reader:
            if num == 0:
                num+=1
                continue
            num+=1
            text = i[1].replace(" ", "")
            if len(text) > maxlen:
                maxlen = len(text)
            lable = i[2]
            onelable = 0

            it = re.finditer(r"[a-zA-Z]+", lable)
            for match in it:
                if match.group()==datalable:
                    onelable=1
            self.labels.append(onelable)
            self.texts.append(text)
        print("maxlen:",maxlen)

    def __getitem__(self, idx):
        text=self.texts[idx]
        label_m=self.labels[idx]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(text,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=256,
                                      return_tensors='pt')  # Return torch.Tensor objects
        # shape [max_len]
        # tensor of token ids  torch.Size([max_len])
        token_ids = encoded_pair['input_ids'].squeeze(0)
        token_ids.cuda()
        # binary tensor with "0" for padded values and "1"  for the other values  torch.Size([max_len])
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        attn_masks.cuda()
        # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens torch.Size([max_len])
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)
        token_type_ids.cuda()
        input_data={
            "token_ids":token_ids,
            "attn_masks":attn_masks,
            "token_type_ids":token_type_ids,
            "label":label_m
        }
        return input_data

    def __len__(self):
        return len(self.texts)
