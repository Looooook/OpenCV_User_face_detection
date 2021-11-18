import torch
from Model import *
from Face_lib import *


def train_one_epoch(epoch):
    total_loss = 0
    # num_counter = 1
    for batch_idx, data in enumerate(train_loader, 0):
        # print(data.shape)
        inputs, target = data
        inputs, target = inputs.cuda(), target.cuda()
        output = net(inputs).cuda()
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 20 == 19:
            # 打印的batch size + 1 是为了显示batch size大小
            print('EPOCH:{},BATCH_IDX:{},LOSS:{}'.format(epoch + 1, batch_idx + 1, total_loss / 20))
            # 在每次loop total_loss都初始为0
            total_loss = 0
            # num_counter += 1


#
# def test():
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, labels in test_loader:
#             data, labels = data.cuda(), labels.cuda()
#             label_hat = net(data).cuda()
#             _, predicted = label_hat.max(dim=1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             print(predicted, labels)
#
#         print('accuracy:{} %,'.format(correct / total * 100))

if __name__ == '__main__':
    net = Model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    train_loader =
