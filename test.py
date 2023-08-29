# import speech_recognition as sr
# import pyttsx3
# import time
# import threading
# import serial
# import QboController
# port = '/dev/serial0'
# ser = serial.Serial(port, baudrate=115200, bytesize = serial.EIGHTBITS, stopbits = serial.STOPBITS_ONE, parity = serial.PARITY_NONE, rtscts = False, dsrdtr =False, timeout = 0)
# QBO = QboController.Controller(ser)
#
# # recog = sr.Recognizer()
# # mic = sr.Microphone()
# # while True:
# #     try:
# #         with mic as source:
# #             audio = recog.listen(source)
# #         command = recog.recognize_google(audio)
# #         print(command)
# #     except:
# #         command = "Unknown !"
# def moveMouth():
#     QBO.SetMouth(0xe1f1f0e)
#     time.sleep(0.05)
#     QBO.SetMouth(0xe0e0e0e)
#     time.sleep(0.05)
#     QBO.SetMouth(0x4040404)
#     time.sleep(0.05)
#     QBO.SetMouth(0x40400)
#     time.sleep(0.05)
#     QBO.SetMouth(0x00000000)
#
#
# def talk():
#     engine = pyttsx3.init()
#     engine.say("Hello, Iam QBO robot")
#     engine.runAndWait()
#
#
# t1 = threading.Thread(target=moveMouth(), args=(10,))
# t2 = threading.Thread(target=talk(), args=(10,))
# t1.start()
# t2.start()
#
#
#
# while t2.is_alive():
#     if not t2.is_alive():
#         break
#     if t2.is_alive():
#         t1.join()
#     if t2.is_alive():
#         t1 = threading.Thread(target=moveMouth, args=())
#     if t2.is_alive():
#         t1.start()
#
#
#
import cv2
import time
import pathlib
cascade_path = pathlib.Path(cv2.__file__).parent.absolute()/"data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))
cap = cv2.VideoCapture(0)
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
center = int(FRAME_WIDTH/2)
centerY = int(FRAME_HEIGHT/2)
head_reposition_factor = 0.2
while True:
    inFace = False
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
    )

    for (x, y, width, height) in faces:
        inFace = True
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 255, 0), 2)
        x_medium = int((x + x + width) / 2)
        y_medium = int((y + y + height) / 2)
        boundingBox = (x,y,width,height)


    cv2.line(img, (x_medium, 0), (x_medium, 480), (0, 255, 0), 2)

    cv2.imshow("Image", img)
    print("X_Default: ", int((center - x_medium) * head_reposition_factor * height / FRAME_HEIGHT))
    print("Y_Default: ", int((center - x_medium) * head_reposition_factor * height / FRAME_HEIGHT))
    key = cv2.waitKey(1)
    if key == 27:
        break
# QBO.SetServo(1, 511, 100)  # Axis,Angle,Speed
# QBO.SetServo(2, 450, 100)  # Axis,Angle,Speed
cap.release()
cv2.destroyAllWindows()