import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import TextDataset
from model import *
import time

from transformers import logging

logging.set_verbosity_warning()
# 加载训练数据
bert_dir = "bert/"
train_dataset = TextDataset()
label_list = {'Love':0, 'Joy':1, 'Anxiety':2, 'Sorrow':3, 'Expect':4, 'Hate':5, 'Surprise':6, 'Anger':7}

train_data_len = len(train_dataset)
print(f"训练集长度：{train_data_len}")


# 创建网络模型
my_model = BertTextModel_last_layer()
my_model=my_model.cuda()
def myeval(pred, tar):

    pred_zero = torch.zeros_like(pred)
    pred_zero[pred > 0.5] = 1
    judge_n=pred_zero.cuda()==tar.cuda()
    error_num=0
    for i in judge_n:
        if False in i:
            error_num+=1
    return error_num,len(judge_n),1-error_num/len(judge_n)

# 优化器
learning_rate = 1e-4
#optimizer = torch.optim.SGD(my_model.parameters(), lr=learning_rate)
#  Adam 参数betas=(0.9, 0.99)
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
# 总共的训练步数
total_train_step = 0
# 总共的测试步数
total_test_step = 0
step = 0
epoch = 1000
# my_model.load_state_dict(torch.load("model/epoch_3"))
writer = SummaryWriter("logs")
test = wandb.init(project="taskNLP", resume="allow")
test.config.update(dict(epoch=50, lr=learning_rate, batch_size=32))

train_loss_his = []
train_totalaccuracy_his = []
test_totalloss_his = []
test_totalaccuracy_his = []
start_time = time.time()
loss_f=nn.MultiLabelSoftMarginLoss(reduction="mean")
train_data_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

for i in range(epoch):
    print(f"-------第{i}轮训练开始-------")
    my_model.train()
    besterr=10000
    allerr=0
    for step, batch_data in enumerate(train_data_loader):
        # writer.add_images("tarin_data", imgs, total_train_step)

        output = my_model(batch_data)
        loss = loss_f(batch_data['label'].cuda(),output)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        train_loss_his.append(loss)
        writer.add_scalar("train_loss", loss, total_train_step)


        errornum,knum,acc = myeval(output, batch_data['label_id'])
        writer.add_scalar("acc_loss", acc, total_train_step)
        print("epoch:",i,"batch:",step,"loss:",loss.data,"acc:",acc)
        test.log({'trainloss': loss.data, 'epoch': i,"step":step,"acc":acc})
        allerr+=errornum
    if allerr<besterr:
        besterr=allerr
        torch.save(my_model.state_dict(), "model/epoch_%d.pth" % i)
