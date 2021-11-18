import cv2

# 加载模型
faceModel = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# 打开摄像头
capture = cv2.VideoCapture(0)
# 获得实时画面
while 1:
    # 读取摄像头数据
    ret, frame = capture.read()
    # print(ret,frame.shape)
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2GRAY)
    # 检测人脸 返回坐标
    faces_location = faceModel.detectMultiScale(gray, scaleFactor=1.2)
    for (x, y, w, h) in faces_location:
        # 画外边框
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
        # 添加名字
        cv2.putText(frame, 'name', (x, y - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0),
                    thickness=2)
        # 显示图片
        cv2.imshow('liveshow', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
