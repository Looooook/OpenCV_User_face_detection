import torch
from PIL import Image
from Model import *
from Face_lib import *
import cv2
import time
import scipy.misc


def frame_to_tensor(frame):
    img_np = np.array(frame)
    # print(img_np.shape)
    img_np = img_np.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img_np).view(1, -1, 100, 100) / 255.0
    return img_tensor


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
    net = Model().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    face_lib_path = './face_dir/'
    face_lib = Face_lib(face_lib_path=face_lib_path)
    train_data_path = './train_data/'
    create_dataset = Create_Dataset(face_lib_path=face_lib_path, train_data_path=train_data_path)
    # create_dataset1 = Create_Dataset()
    # 加载模型
    face_model = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    goal = int(input('录入人脸数据0 or 人脸识别1'))

    # 打开摄像头
    capture = cv2.VideoCapture(0)
    if goal == 0:
        name_in = input('请输入姓名：')

        # 读取读片
        img_counter = 0
        face_lib.create_file(name_in)
        while 1:
            ret, frame = capture.read()
            # print(type(frame))
            # 初始化图片
            grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # 人脸识别
            face_location = face_model.detectMultiScale(grey, scaleFactor=1.2, minSize=(200, 200))
            for (x, y, w, h) in face_location:
                cv2.rectangle(frame, pt1=(x - 10, y - 10), pt2=(x + w + 10, y + h + 10), color=(0, 0, 255))
                cv2.imshow('face_detection', frame)
                # while and if difference
                if img_counter < 10:
                    img = Image.fromarray(frame)
                    time.sleep(1)
                    face_lib.save_face_img(name_in, img)
                    img_counter += 1
                    print(img_counter)

            if cv2.waitKey(1) & 0xff == ord('q'):
                face_lib.rename()
                break
        capture.release()
        cv2.destroyAllWindows()
    elif goal == 1:
        print('Now training!')
        create_dataset.clear_dataset()
        create_dataset.resize()
        train_loader = create_dataset.img_to_loader(train_img_path=train_data_path, batch_size=10, shuffle=True)
        category = '7'
        for i in range(25):
            train_one_epoch(i)
        print('------training over------')
        while 1:
            ret, frame = capture.read()
            # 初始化图片
            grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # frame = frame
            # 人脸识别
            face_location = face_model.detectMultiScale(grey, scaleFactor=1.2, minSize=(200, 200))
            for (x, y, w, h) in face_location:
                cv2.rectangle(frame, pt1=(x - 10, y - 10), pt2=(x + w + 10, y + h + 10), color=(255, 0, 0))
                img = Image.fromarray(frame).resize((100, 100), Image.BILINEAR)
                img_tensor = frame_to_tensor(img).cuda()
                predicted = net(img_tensor)
                predicted_str = str(predicted.sum().item())  # 1.0
                # print(predicted_str)

                for sub_file in os.listdir(face_lib_path):
                    # print(sub_file)
                    # print(type(sub_file.split('_')[0]),type(predicted_str))
                    if int(sub_file.split('_')[0]) == int(float(predicted_str)):
                        # category = sub_file.split('_')[1]
                        category = sub_file.split('_')[1]

                cv2.putText(frame, category, (x, y - 10), color=(0, 255, 0), thickness=3,
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1)
                cv2.imshow('face_detection', frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()

# 在原图像上显示 框 & 人名
