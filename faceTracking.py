import cv2
import pathlib
import serial #handles the serial ports
import time
import QboController
import speech_recognition as sr
import keyboard
import threading
import pyttsx3
cascade_path = pathlib.Path(cv2.__file__).parent.absolute()/"data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))
# port = '/dev/serial0'
# ser = serial.Serial(port, baudrate=115200, bytesize = serial.EIGHTBITS, stopbits = serial.STOPBITS_ONE, parity = serial.PARITY_NONE, rtscts = False, dsrdtr =False, timeout = 0)
# QBO = QboController.Controller(ser)

#Init Frame
cap = cv2.VideoCapture(0)
#Highest Resolution
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
center = int(FRAME_WIDTH/2)
centerY = int(FRAME_HEIGHT/2)
# QBO.SetServo(1, xDefault, 100)#Axis,Angle,Speed (X Axis)
# QBO.SetServo(2, yDefault, 100)#Axis,Angle,Speed (Y Axis)
recog = sr.Recognizer()
mic = sr.Microphone()

class QBO_Function():
    def initPosition(self, x_current, y_current):
        X_MAX = 725
        X_MIN = 290
        Y_MAX = 550
        Y_MIN = 420
        if x_current > X_MAX:
            x_current = X_MAX
        if x_current < X_MIN:
            x_current = X_MIN
        if y_current > Y_MAX:
            y_current = Y_MAX
        if y_current < Y_MIN:
            y_current = Y_MIN
        #QBO.SetServo(1, x_current, 100)
        time.sleep(0.5)
        #QBO.SetServo(2, y_current, 100)
        time.sleep(0.5)
        #QBO.SetPid(1, 26, 2, 16)  # Set PID horizontal servo
        time.sleep(0.5)
        #QBO.SetPid(2, 26, 2, 16)  # Set PID vertical servo
        time.sleep(0.5)
        return x_current, y_current

    def headFollowing(self, xHead, yHead):
        #command = audio.listen()
        X_MAX = 725
        X_MIN = 290
        Y_MAX = 550
        Y_MIN = 420
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

            try:
                for (x, y, width, height) in faces:
                    inFace = True
                    cv2.rectangle(img, (x, y), (x + width, y + height), (255, 255, 0), 2)
                    x_medium = int((x + x + width) / 2)
                    y_medium = int((y + y + height) / 2)
                cv2.line(img, (x_medium, 0), (x_medium, 480), (0, 255, 0), 2)
                cv2.line(img, (center, 0), (center, 480), (255, 255, 0), 2)
            except:
                print("No Face Detected !")
                img = cv2.putText(img, "NO FACE !", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("Faces", img)
            if inFace == False:
                # QBO.SetServo(1, xDefault, 100)
                # QBO.SetNoseColor(0)
                print("No Face Detected !")
                img = cv2.putText(img, 'NO FACE !', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("Faces", img)
                continue
            # QBO.SetNoseColor(1)

            cv2.imshow("Faces", img)
            if x_medium < center + 3:
                xHead = xHead + int((center - x_medium) * head_reposition_factor * height/FRAME_HEIGHT)
            elif x_medium > center - 3:
                xHead = xHead - int((center - x_medium) * head_reposition_factor * height/FRAME_HEIGHT)
            xHead = X_MIN if xHead < X_MIN else X_MAX if xHead > X_MAX else xHead
            if y_medium > centerY + 3:
                yHead = yHead + int((center - x_medium) * head_reposition_factor * height/FRAME_HEIGHT)
            elif y_medium < centerY - 3:
                yHead = yHead - int((center - x_medium) * head_reposition_factor * height/FRAME_HEIGHT)
            yHead = Y_MIN if yHead < Y_MIN else Y_MAX if yHead > Y_MAX else yHead
            # QBO.SetServo(1, xHead, 100)
            # QBO.SetServo(2, yHead, 100)
            # QBO.SetPid(1, 26, 2, 16)  # Set PID horizontal servo
            # QBO.SetPid(2, 26, 2, 16)  # Set PID horizontal servo

            print("Test: ", x_medium)
            key = cv2.waitKey(1)
            if key == 27:
                break
        # QBO.SetServo(1, 511, 100)  # Axis,Angle,Speed
        # QBO.SetServo(2, 450, 100)  # Axis,Angle,Speed
        cap.release()
        cv2.destroyAllWindows()

    def getCommand(self):
        try:
            with mic as source:
                audio = recog.listen(source)
            command = recog.recognize_google(audio)
            print(command)
        except:
            command = "Can't hear !"
        return command

    def voiceCondition(self, command):
        robotCommand = "robot " + command
        return robotCommand

# class QBO_Speak():
#     def moveMouth(self):
#         print("Task 1")
#         time.sleep(0.3)
#         print("Task 1.1")
#         time.sleep(0.3)
#
#     def robotCommand(self):
#         engine = pyttsx3.init()
#         engine.say("Hello, this is QBO robot")
#         engine.runAndWait()


robot = QBO_Function()
#robotTalk = QBO_Speak()
x_current, y_current = robot.initPosition(511, 400)

def moveMouth():
    print("Task 1")
    time.sleep(0.3)
    print("Task 1.1")
    time.sleep(0.3)


def talk():
    engine = pyttsx3.init()
    engine.say("Hello, Iam QBO robot")
    engine.runAndWait()

    # print("Task 2")
    # time.sleep(0.3)
    # print("Task 2.1")
    # time.sleep(0.3)
    # print("Task 2.2")
    # time.sleep(0.3)
# if __name__ == "__main__":
#     t1 = threading.Thread(target=moveMouth(), args=(10,))
#     t2 = threading.Thread(target=talk(), args=(10,))
#     t1.start()
#     t2.start()
# moveMouth()
# talk()

t1 = threading.Thread(target=moveMouth, args=())
t1.start()
t2 = threading.Thread(target=talk, args=())
t2.start()
while t2.is_alive():
    if not t2.is_alive():
        break
    if t2.is_alive():
        t1.join()
    if t2.is_alive():
        t1 = threading.Thread(target=moveMouth, args=())
    if t2.is_alive():
        t1.start()


while True:
    # #command = audio.listen()
    # inFace = False
    # success, img = cap.read()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = clf.detectMultiScale(
    #     gray,
    #     scaleFactor=1.3,
    #     minNeighbors=5,
    # )
    #
    # try:
    #     for (x,y,width,height) in faces:
    #         inFace = True
    #         cv2.rectangle(img,(x,y),(x+width, y+height), (255,255,0),2)
    #         x_medium = int((x + x + width)/2)
    #
    #     cv2.line(img, (x_medium, 0), (x_medium, 480), (0,255,0), 2)
    #     cv2.line(img, (center, 0), (center, 480), (255, 255, 0), 2)
    # except:
    #     print("No Face Detected !")
    #     img = cv2.putText(img, "NO FACE !", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    #     cv2.imshow("Faces", img)
    # if inFace == False:
    #     #QBO.SetServo(1, xDefault, 100)
    #     #QBO.SetNoseColor(0)
    #     print("No Face Detected !")
    #     img = cv2.putText(img, 'NO FACE !', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    #     cv2.imshow("Faces", img)s
    #     continue
    # #QBO.SetNoseColor(1)
    #
    #
    # #Start

    robot.headFollowing(x_current, y_current)
    # mouth = robotTalk.moveMouth()
    # text = robotTalk.robotCommand("Hello, this is testing")
    # robotTalk.robotTalk(mouth, text)
    #command = robot.getCommand()
    # if command == robot.voiceCondition("follow"):
    #     robot.headFollowing(x_current, y_current)
    # if command == robot.voiceCondition("repete"):
    #     print("Break")




