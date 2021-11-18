import cv2

# 加载模型
face_model = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# 打开摄像头
capture = cv2.VideoCapture(0)

while 1:
    # 得到图像
    ret, frame = capture.read()
    # 处理图像
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 人脸识别
    face_location = face_model.detectMultiScale(grey)
    # 框选+字
    for (x, y, w, h) in face_location:
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=3)
        cv2.putText(frame, 'ZT', (x, y - 7), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0),
                    thickness=2)
        # 显示图像
        cv2.imshow('face_detection', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
