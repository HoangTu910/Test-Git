#!/usr/bin/env python3

###############################################################
# Qbo_face_detect_and_head_track.py
#
# Based upon "Face Detection in Python Using a Webcam"
# (https://realpython.com/face-detection-in-python-using-a-webcam/),
# and "OpenCV Object Tracking"
# (https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/)
# and "Increasing webcam FPS with Python and OpenCV"
# (https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/)
# and "OpenQbo v3.0" (https://gitlab.com/thecorpora/QBO)
#
# 1. Connect to Q.bo using VNC
# 2. Copy this file Qbo_face_detect_and_track.py to the /home/pi/ directory
# 3. Press Ctrl+Alt+T to open a terminal
# 4. Enter
#    sudo pkill python
#    sudo pkill chromium
# 5. Enter
#    sudo apt-get install libjasper-dev
#    sudo apt-get install libqtgui4
#    sudo apt-get install libqt4-test
#    sudo pip3 install pyyaml
#    sudo pip3 install imutils
#    sudo pip3 install opencv-contrib-python
# 6. Enter
#    python3 Qbo_face_detect_and_head_track.py
#
# Press 'q' to quit.
#
# We need to have a recent OpenCV available for Python3 because of the used tracker CSRT
# or 'Discriminative Correlation Filter (with Channel and Spatial Reliability)'
# which requires minimum OpenCV 3.4.2)
###############################################################

# Allow loading libraries from /opt/qbo/
import sys

sys.path.insert(0, '/opt/qbo/')

# Import all other required system librariess
from imutils.video import FPS
import logging as log
import imutils
import serial
import yaml
import time
import cv2

# Import Qbo libraries
import QboController as Controller
controller = Controller.Controller
# Initialize logging
log.basicConfig(
    filename='Qbo_face_detect_and_follow.log',
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log.info('Initialized logging')

###############################################################
# Q.bo global constants
###############################################################

# Load configuration
log.info('Reading /opt/qbo/config.yml')
QBO_CONFIG = yaml.safe_load(open("/opt/qbo/config.yml"))

# Vision
# Highest resolution that tested out well
VISION_FRAME_WIDTH = 320
VISION_FRAME_WIDTH_MIDDLE = int(VISION_FRAME_WIDTH / 2)  # Middle of the frame, calculated one
VISION_FRAME_HEIGHT = 240
VISION_FRAME_HEIGHT_MIDDLE = int(VISION_FRAME_HEIGHT / 2)  # Middle of the frame, calculated one
VISION_MAX_FACE_TRACKING_FRAMES = 10  # 10 frames (about 10 seconds) tracking then redetect face

## Head
HEAD_X_MAX = 725  # Head X maximum
HEAD_X_MIN = 290  # Head X minimum
HEAD_Y_MAX = 550  # Head Y maximum
HEAD_Y_MIN = 420  # Head Y minimum
HEAD_REPOSITION_FACTOR = 0.2  # Vison to head movement ratio


###############################################################
# Q.bo functions and classes
###############################################################

def head_reposition():
    global head_x_coor, head_y_coor, head_reposition_mutex, face_found

    if (head_reposition_mutex):
        return  # We're already moving the head, we need to be exclusive
    head_reposition_mutex = True  # We're going to move the head, set exclusive

    log.info("Repositioning head: head_x_coor " + str(head_x_coor) + ", head_y_coor " + str(head_y_coor))
    controller.SetServo(1, head_x_coor, int(QBO_CONFIG["servoSpeed"]))  # Move back to original horizontal position
    time.sleep(0.2)  # Pause for next command
    controller.SetServo(2, head_y_coor, int(QBO_CONFIG["servoSpeed"]))  # Move back to orginal vertical position

    head_reposition_mutex = False  # We're done moving the head, unset exclusive

    return


def head_original_position():
    global head_x_coor, head_y_coor

    head_x_coor = 511

    # With vertical head position adjustment (based on extra parameter in config.yml)
    head_y_coor = int(
        HEAD_Y_MIN
        + float(QBO_CONFIG["headYPosition"])
        / 100
        * (HEAD_Y_MAX - HEAD_Y_MIN)
    )  # Vertically, depending on a configuration value set the head high, medium or down
    log.info("Calculated initial head position: head_x_coor " + str(head_x_coor) + ", head_y_coor " + str(head_y_coor))

    head_reposition()

    return


###############################################################
# Q.bo state
###############################################################

# Vision
vision_face_detected_time = 0  # Last time a face was detected
vision_face_tracking_frames = 0  # Are we tracking a face? If yes, how many more frames to track without redetection
vision_face_tracker = None  # Place holder for face tracker once detected
vision_face_bounding_box = None  # Place holder for face bounding box (rectangle) if found
vision_face_frontal_detection = True  # Start by detecting a frontal face
vision_face_detected_time = 0  # First time we detected a face while tracking
vision_face_not_detected_time = 0  # First time we didn't detect a face anymore

# Head
head_reposition_mutex = False
head_x_coor = 511  # Horizontally in the middle

# Without vertical head position adjustment

head_y_coor = 450

# With vertical head position adjustment (based on extra parameter in config.yml)

# head_y_coor = int(
#     HEAD_Y_MIN
#     + float(QBO_CONFIG["headYPosition"])
#     / 100
#     * (HEAD_Y_MAX - HEAD_Y_MIN)
# )                                         # Vertically, depending on a configuration value set the head high, medium or down
# log.info("Calculated initial head position: head_x_coor " + str(head_x_coor) + ", head_y_coor " + str(head_y_coor))


###############################################################
# Q.bo code
###############################################################

# Initialize controller
if len(sys.argv) > 1:
    port = sys.argv[1]
else:
    port = "/dev/serial0"
try:
    ser = serial.Serial(
        port,
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        stopbits=serial.STOPBITS_ONE,
        parity=serial.PARITY_NONE,
        rtscts=False,
        dsrdtr=False,
        timeout=0)
    log.info("Open serial port sucessfully.")
    log.info(ser.name)
except:
    log.error("Error opening serial port.")
    exit(1)
controller = Controller(ser)

# Initialize head
log.info("Positioning head: head_x_coor " + str(head_x_coor) + ", head_y_coor " + str(head_y_coor))
controller.SetServo(1, head_x_coor, int(QBO_CONFIG["servoSpeed"]))
time.sleep(0.5)
controller.SetServo(2, head_y_coor, int(QBO_CONFIG["servoSpeed"]))
time.sleep(0.5)
controller.SetPid(1, 26, 2, 16)  # Set PID horizontal servo
time.sleep(0.5)
controller.SetPid(2, 26, 2, 16)  # Set PID vertical servo
time.sleep(0.5)

# Initialize nose
controller.SetNoseColor(0)  # Off QBO nose brigth

# Initialize vision

# Get ready to start getting images from the webcam
log.info('Starting the webcam thread')
vision_webcam = cv2.VideoCapture(int(QBO_CONFIG['camera']))  # Initialize webcam on configured source
vision_webcam.set(cv2.CAP_PROP_FRAME_WIDTH, VISION_FRAME_WIDTH)  # Set webcam frame width
vision_webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, VISION_FRAME_HEIGHT)  # Set webcam frame height

# Allow the camera to startup
time.sleep(2)

# Start the FPS counter
log.info('Starting the frames per second counter')
vision_fps = FPS().start()

# Setup face detectors (frontal face detection and pofile face detection)
log.info('Setup cascade classifier frontal face detection (/opt/qbo/haarcascades/haarcascade_frontalface_alt2.xml)')
vision_face_frontal = cv2.CascadeClassifier(
    '/opt/qbo/haarcascades/haarcascade_frontalface_alt2.xml')  # frontal face pattern detection
log.info('Setup cascade classifier profile face detection (/opt/qbo/haarcascades/haarcascade_profileface.xml)')
vision_face_profile = cv2.CascadeClassifier(
    '/opt/qbo/haarcascades/haarcascade_profileface.xml')  # side face pattern detection

# Initialization face detection
vision_face_tracking_frames = 0  # Are we tracking a face? If yes, how many more frames to track without redetection
vision_face_tracker = None  # Place holder for face tracker once detected
vision_face_bounding_box = None  # Place holder for face bounding box (rectangle) if found
vision_face_frontal_detection = True  # Start by detecting a frontal face

# Main endless loop
# DEBUG: Display time since last run
qbo_last_time_run = 0
while True:
    # DEBUG: Display time since last run
    log.debug("Frame time: " + str(time.time() - qbo_last_time_run))
    qbo_last_time_run = time.time()

    # Get the next frame
    successful, vision_frame = vision_webcam.read()
    if not successful:
        log.error('Not successful grabbing a frame, skipping and try again')
        pass

    # Aren't we tracking a face already?
    if not vision_face_tracking_frames:

        # We are not tracking a face, try detecting a face

        # For now, we are only interested in the 'largest'
        # face, and we determine this based on the largest
        # area of the found bounding_box. First initialize the
        # required variables to 0
        vision_face_max_area = 0
        vision_face_bounding_box = None

        # Convert to gray for faster processing
        vision_frame_gray = cv2.cvtColor(vision_frame, cv2.COLOR_BGR2GRAY)

        if vision_face_frontal_detection:
            # Detect a frontal face
            log.debug('Detecting a frontal face')
            faces = vision_face_frontal.detectMultiScale(
                vision_frame_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

        else:
            # Detect a profile face
            log.debug('Detecting a profile face')
            faces = vision_face_profile.detectMultiScale(
                vision_frame_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

        # Did we detect frontal faces ...
        if not len(faces):
            # .... no, try another face detection next time
            vision_face_frontal_detection = not vision_face_frontal_detection
            # ... and make sure it is seen Q.bo didn't detect a face, set nose black
            log.debug("We're not detecting a face.")
            controller.SetNoseColor(0)

            # We don't see a face, set parameters
            vision_face_detected_time = 0
            if not vision_face_not_detected_time:
                vision_face_not_detected_time = time.time()
        else:
            # ... yes, log
            log.debug('Face(s) found')

            # ... and make sure it is seen Q.bo detected a face, set nose blue
            controller.SetNoseColor(1)

            # Loop over all faces and check if the area for this
            # face is the largest so far
            for (_x, _y, _w, _h) in faces:
                if _w * _h > vision_face_max_area:
                    vision_face_max_area = _w * _h
                    vision_face_bounding_box = (_x, _y, _w, _h)

            # Initialize the tracker on the largest face in the picture
            if vision_face_max_area > 0:
                # We detected a face, set parameters
                if not vision_face_detected_time:
                    vision_face_detected_time = time.time()
                vision_face_not_detected_time = 0

                # Initialize face we are going to track
                log.debug('Initialize face tracker')
                # Experiment between
                #     cv2.TrackerCSRT_create        # Tends to be more accurate than KCF but slightly slower.
                #     cv2.TrackerKCF_create         # Faster than BOOSTING and MIL. Similar to MIL and KCF, does not handle full occlusion well.
                #     cv2.TrackerBoosting_create    # This tracker is slow and doesnâ€™t work very well.
                #     cv2.TrackerMIL_create         # Better accuracy than BOOSTING tracker but does a poor job of reporting failure.
                #     cv2.TrackerTLD_create         # Incredibly prone to false-positives.
                #     cv2.TrackerMedianFlow_create  # Does a nice job reporting failures; however, if there is too large of a jump in motion, the model will fail.
                #     cv2.TrackerMOSSE_create       # Very, very fast. Not as accurate as CSRT or KCF but a good choice if you need pure speed.
                vision_face_tracker = cv2.TrackerCSRT_create()
                if vision_face_tracker.init(vision_frame, vision_face_bounding_box):
                    vision_face_tracking_frames = VISION_MAX_FACE_TRACKING_FRAMES
            else:
                # We did not detect a face, set parameters
                if not vision_face_not_detected_time:
                    vision_face_not_detected_time = time.time()
                vision_face_detected_time = 0

    else:

        # We are tracking a face!

        # Get the new bounding_box of the object
        log.debug('Trying to track detected face')
        (successful, vision_face_bounding_box) = vision_face_tracker.update(vision_frame)

        # If a face is detected draw a green bounding_box on the frame
        if successful:
            # Calculate remainder of frames to track before automatic redetection
            vision_face_tracking_frames = vision_face_tracking_frames - 1

            # Draw a green bounding_box around detected face
            log.debug(
                'Tracked the face, ' + str(vision_face_tracking_frames) + ' frames to track before auto-redetection')
            (_x, _y, _w, _h) = [int(_v) for _v in vision_face_bounding_box]
            cv2.rectangle(vision_frame, (_x, _y), (_x + _w, _y + _h), (0, 255, 0), 2)
        else:
            # We lost the face, redetect
            vision_face_tracking_frames = 0

    # Head

    # If we're seeing a face and not touched...
    log.debug("Deciding if we have to move the head")
    # if vision_face_tracking_frames and not touch_sensor:
    if vision_face_tracking_frames:
        # ... move the head towards the face

        # Calculate the central point of the face
        (_x, _y, _w, _h) = [int(_v) for _v in vision_face_bounding_box]
        vision_face_central_point_x = _x + int(_w / 2)
        vision_face_central_point_y = _y + int(_h / 2)
        vision_face_area_percentage = _h / VISION_FRAME_HEIGHT

        # Show vision face central point
        cv2.rectangle(vision_frame, (vision_face_central_point_x - 2, vision_face_central_point_y - 2),
                      (vision_face_central_point_x + 2, vision_face_central_point_y + 2), (0, 255, 0), 2)

        # Alternative 1: Fixed head repositioning

        # Initialize test to show
        movement_text = "face"

        # Attempt with movement towards face with fixed steps
        if vision_face_central_point_x > VISION_FRAME_WIDTH_MIDDLE + 3:
            # Turn left
            movement_text = movement_text + ", left"
            head_x_coor = head_x_coor - int((
                                                        vision_face_central_point_x - VISION_FRAME_WIDTH_MIDDLE) * HEAD_REPOSITION_FACTOR * vision_face_area_percentage)
        elif vision_face_central_point_x < VISION_FRAME_WIDTH_MIDDLE - 3:
            # Turn right
            movement_text = movement_text + ", right"
            head_x_coor = head_x_coor + int((
                                                        VISION_FRAME_WIDTH_MIDDLE - vision_face_central_point_x) * HEAD_REPOSITION_FACTOR * vision_face_area_percentage)
        head_x_coor = HEAD_X_MIN if head_x_coor < HEAD_X_MIN else HEAD_X_MAX if head_x_coor > HEAD_X_MAX else head_x_coor

        if vision_face_central_point_y > VISION_FRAME_HEIGHT_MIDDLE + 3:
            # Look up
            movement_text = movement_text + ", down"
            head_y_coor = head_y_coor + int((
                                                        vision_face_central_point_y - VISION_FRAME_HEIGHT_MIDDLE) * HEAD_REPOSITION_FACTOR * vision_face_area_percentage)
        elif vision_face_central_point_y < VISION_FRAME_HEIGHT_MIDDLE - 3:
            # Look down
            movement_text = movement_text + ", up"
            head_y_coor = head_y_coor - int((
                                                        VISION_FRAME_HEIGHT_MIDDLE - vision_face_central_point_y) * HEAD_REPOSITION_FACTOR * vision_face_area_percentage)
        head_y_coor = HEAD_Y_MIN if head_y_coor < HEAD_Y_MIN else HEAD_Y_MAX if head_y_coor > HEAD_Y_MAX else head_y_coor

        # Show movement text
        (_x, _y, _w, _h) = [int(_v) for _v in vision_face_bounding_box]
        _y = _y - 15 if _y - 15 > 15 else _y + 15
        cv2.putText(vision_frame, movement_text, (_x, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        log.debug("Reposition head " + str(head_x_coor) + str(head_y_coor))
        head_reposition()

        # Alternative 2: Fixed head repositioning

        # head_offset_x = vision_face_central_point_x - VISION_FRAME_WIDTH_MIDDLE)  # Right is more offset
        # head_x_coor = head_x_coor + int(head_offset_x * HEAD_REPOSITION_FACTOR)
        # head_x_coor = HEAD_X_MIN if head_x_coor < HEAD_X_MIN else HEAD_X_MAX if head_x_coor > HEAD_X_MAX else head_x_coor

        # head_offset_y = VISION_FRAME_HEIGHT_MIDDLE - vision_face_central_point_y  # Up is more offset
        # head_y_coor = head_y_coor + int(head_offset_y * HEAD_REPOSITION_FACTOR)
        # head_y_coor = HEAD_Y_MIN if head_y_coor < HEAD_Y_MIN else HEAD_Y_MAX if head_y_coor > HEAD_Y_MAX else head_y_coor

        # log.debug("Reposition head " + str(head_x_coor) + str(head_y_coor))
        # head_reposition()
        # _thread.start_new_thread(head_reposition, ())  # Reposition head

        # Alternative 3: Relative head repositioning

        # head_offset_x = VISION_FRAME_WIDTH_MIDDLE - vision_face_central_point_x  # Left is more offset
        # log.debug("Face seen, head relative x: " + str(head_offset_x))
        # if (head_offset_x > 20) | (head_offset_x < -20):
        #     time.sleep(0.002)
        #     controller.SetAngleRelative(1, int(head_offset_x >> 1))
        #     # wait for move
        #     time.sleep(0.05)

        # head_offset_y = VISION_FRAME_HEIGHT_MIDDLE - vision_face_central_point_y  # Up is more offset
        # log.debug("Face seen, head relative y: " + str(head_offset_y))
        # if (head_offset_y > 10) | (head_offset_y < -10):
        #     time.sleep(0.002)
        #     controller.SetAngleRelative(2, int(head_offset_y >> 1))
        #     # wait for move
        #     time.sleep(0.05)

        # Alternative 4: Relative head repositioning

        # head_offset_x = VISION_FRAME_WIDTH_MIDDLE - vision_face_central_point_x  # Left is more offset
        # log.debug("Face seen, head relative x: " + str(head_offset_x))
        # time.sleep(0.002)
        # log.debug("SetAngleRelative(1, " + str(int(head_offset_x/5)) + ")")
        # controller.SetAngleRelative(1, int(head_offset_x/5))
        # # wait for move
        # time.sleep(0.1)

        # head_offset_y = VISION_FRAME_HEIGHT_MIDDLE + vision_face_central_point_y  # Up is more offset
        # log.debug("Face seen, head relative y: " + str(head_offset_y))
        # time.sleep(0.002)
        # log.debug("SetAngleRelative(2, " + str(int(head_offset_y/5)) + ")")
        # controller.SetAngleRelative(2, int(head_offset_y/5))
        # # wait for move
        # time.sleep(0.1)

    elif vision_face_not_detected_time - time.time() > 10:
        head_original_position()

    # Display the resulting frame
    cv2.imshow('Video', vision_frame)

    # If the 'q' key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        log.info('Stopping because the q key was pressed')
        break

    # Update the FPS counter
    vision_fps.update()

# Stop the timer and display FPS information
vision_fps.stop()
log.info('Elapsed time: {:.2f}'.format(vision_fps.elapsed()))
log.info('Vision approx. FPS: {:.2f}'.format(vision_fps.fps()))

# When everything is done, release the capture
cv2.destroyAllWindows()

log.info('Script ended')