import csv
import re

import logging

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

label_m={'Love':0, 'Joy':1, 'Anxiety':2, 'Sorrow':3, 'Expect':4, 'Hate':5, 'Surprise':6, 'Anger':7}
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
                onelable[label_m[match.group()]]=1
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
