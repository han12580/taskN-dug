import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from dataset import TextDataset, TextDataset_last, OneTextDataset
from model import *
import time
from transformers import logging
logging.set_verbosity_warning()
# 加载训练数据
bert_dir = "bert/"
train_dataset = OneTextDataset("Love")

train_data_len = len(train_dataset)
print(f"训练集长度：{train_data_len}")

numClass=4
# 创建网络模型
my_model = bertclassify()
my_model=my_model.cuda()

my_model_encoder=text_decoder().cuda()

def loss_F(pred,tar):
    return torch.mean(torch.abs(pred-tar)**2)

def myeval(pred, tar):
    predlabel = torch.argmax(pred,dim=1)


# 优化器
learning_rate = 5e-3

# onemodel.load_state_dict(torch.load("model/decoder_%d.pth"%index))
optimizer = torch.optim.Adam(my_model_encoder.parameters(), lr=learning_rate, betas=(0.9, 0.99))

# 总共的训练步数
total_train_step = 0
# 总共的测试步数
total_test_step = 0
step = 0
epoch = 500

writer = SummaryWriter("logs")
# test = wandb.init(project="taskNLP", resume="allow")
# test.config.update(dict(epoch=500, lr=learning_rate, batch_size=256))


train_loss_his = []
train_totalaccuracy_his = []
test_totalloss_his = []
test_totalaccuracy_his = []
start_time = time.time()
all_k_step=0
best_acc=0
loss_f=nn.CrossEntropyLoss()
# loss_f=nn.BCELoss()
for i in range(epoch):
    print(f"-------第{i}轮训练开始-------")


    allerrornum_test=0
    allnum_test=0
    allerrornum_train=0
    allnum_train=0
    from sklearn.model_selection import KFold

    all_k_step = 0
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, val_index in kf.split(train_dataset):
        print("-----------train----------")
        all_k_step+=1
        train_fold = torch.utils.data.dataset.Subset(train_dataset, train_index)
        val_fold = torch.utils.data.dataset.Subset(train_dataset, val_index)

        train_data_loader = DataLoader(dataset=train_fold, batch_size=32, shuffle=True)
        val_data_loader = DataLoader(dataset=val_fold, batch_size=32, shuffle=True)

        total_train_loss = 0
        for step, batch_data in enumerate(train_data_loader):
            de_onemodel = my_model(batch_data)
            all_classouts=list()
            all_tars=list()
            allloss=0
            # writer.add_images("tarin_data", imgs, total_train_step)

            my_model_encoder.train()
            output = my_model_encoder(de_onemodel)
            all_classouts.append(output)
            labels=batch_data['label']

            loss = loss_f(output, labels)
            print("loss",loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_step = total_train_step + 1
