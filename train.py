import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
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

numClass=8
# 创建网络模型
my_model = bertclassify(1)
my_model=my_model.cuda()

my_models=[text_decoder().cuda(),
           text_decoder().cuda(),
           text_decoder().cuda(),
           text_decoder().cuda(),
           text_decoder().cuda(),
           text_decoder().cuda(),
           text_decoder().cuda(),
           text_decoder().cuda()]

def loss_F(pred,tar):
    return torch.mean(torch.abs(pred-tar)**2)

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
learning_rate = 5e-3
optimizers=[]
for onemodel in my_models:
    optimizer = torch.optim.Adam(onemodel.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    optimizers.append(optimizer)
# 总共的训练步数
total_train_step = 0
# 总共的测试步数
total_test_step = 0
step = 0
epoch = 500

writer = SummaryWriter("logs")
test = wandb.init(project="taskNLP", resume="allow")
test.config.update(dict(epoch=500, lr=learning_rate, batch_size=256))


train_loss_his = []
train_totalaccuracy_his = []
test_totalloss_his = []
test_totalaccuracy_his = []
start_time = time.time()
all_k_step=0
best_acc=0
loss_f=nn.MultiLabelMarginLoss(reduction="mean")
# loss_f=nn.BCELoss()
for i in range(epoch):
    print(f"-------第{i}轮训练开始-------")


    allerrornum_test=0
    allnum_test=0
    allerrornum_train=0
    allnum_train=0
    from sklearn.model_selection import KFold

    all_k_step = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, val_index in kf.split(train_dataset):
        print("-----------train----------")
        all_k_step+=1
        train_fold = torch.utils.data.dataset.Subset(train_dataset, train_index)
        val_fold = torch.utils.data.dataset.Subset(train_dataset, val_index)

        train_data_loader = DataLoader(dataset=train_fold, batch_size=256, shuffle=True)
        val_data_loader = DataLoader(dataset=val_fold, batch_size=256, shuffle=True)

        total_train_loss = 0
        for step, batch_data in enumerate(train_data_loader):
            de_onemodel = my_model(batch_data)
            all_classouts=list()
            all_tars=list()
            allloss=0
            # writer.add_images("tarin_data", imgs, total_train_step)
            for idclass in range(numClass):
                my_models[idclass].train()
                output = my_models[idclass](de_onemodel)
                all_classouts.append(output)
                tar=torch.unsqueeze(batch_data['label'][:,idclass].cuda(),1)
                all_tars.append(tar)
                loss = loss_F(output,tar.long())
                optimizers[idclass].zero_grad()
                loss.backward()
                optimizers[idclass].step()
                total_train_step = total_train_step + 1



                writer.add_scalar("train_loss", loss, total_train_step)
                allloss+=loss.data

            all_classouts=torch.cat(all_classouts,1)
            all_tars=torch.cat(all_tars,1)
            errornum,knum,acc = myeval(all_classouts, all_tars)
            allnum_train+=knum
            allerrornum_train+=errornum
            writer.add_scalar("step_acc", acc, total_train_step)
            print("epoch:",i,"k_step",all_k_step,"batch:",step,"loss:",allloss,"acc:",acc)
            test.log({"total_train_step":total_train_step,"acc":acc})
        writer.add_scalar("train_loss_k", total_train_loss, all_k_step)

        # 测试开始
        total_test_loss = 0
        my_model.eval()
        print("----------eval----------")
        test_total_accuracy = 0
        for step, batch_data in enumerate(val_data_loader):
            de_onemodel = my_model(batch_data)
            all_classouts=list()
            all_tars=list()
            allloss=0
            # writer.add_images("tarin_data", imgs, total_train_step)
            for idclass in range(numClass):
                my_models[idclass].eval()
                output = my_models[idclass](de_onemodel)
                all_classouts.append(output)
                tar=torch.unsqueeze(batch_data['label'][:,idclass].cuda(),1)
                all_tars.append(tar)
                loss = loss_F(output,tar.long())
                total_train_step = total_train_step + 1
                writer.add_scalar("test_loss", loss, total_train_step)
                allloss+=loss.data

            all_classouts=torch.cat(all_classouts,1)
            all_tars=torch.cat(all_tars,1)
            errornum,knum,acc = myeval(all_classouts, all_tars)
            allnum_test+=knum
            allerrornum_test+=errornum
        print( "acc", 1-allerrornum_test / allnum_test,"total_test_loss",total_test_loss)
        test.log({"epoch_test_acc": 1-allerrornum_test / allnum_test, "epoch":i})
        writer.add_scalar("test_loss_k", total_test_loss, all_k_step)
    if best_acc<1-allerrornum_test / allnum_test:
        model_dir="epoch_%d/"%i
        os.mkdir("model/"+model_dir)
        for model_id,model_m in enumerate(my_models):
            torch.save(model_m.state_dict(), "model/"+model_dir+"decoder_%d.pth" % model_id)
        torch.save(my_model.state_dict(), "model/"+model_dir+"encoder.pth")
        best_acc=1-allerrornum_test / allnum_test
    writer.add_scalar("epoch_train_acc", 1-allerrornum_train/allnum_train, i)
    writer.add_scalar("epoch_test_acc", 1-allerrornum_test / allnum_test, i)
    # test.log({'epoch': i, "train_acc": 1-allerrornum_train/allnum_train})
    # test.log({"epoch_test_acc": 1-allerrornum_test / allnum_test, "epoch":i})