
# EXPERIMENTAL CREW DETECTION

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import face_recognition
import io
import time
import cv2
import numpy as np
import datetime
import sys
import json
import requests
import os

from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

global isCrew
isCrew = False

global serial
serial = 0

global person
person = 0

global eye
eye = 0

global firstRun
firstRun = True

global known_face_encodings
known_face_encodings = []

global previous_image
previous_image = cv2.imread('./images/champagne.jpg')

app = ClarifaiApp("6aN53theduqKsp02t7pC4vtBHD6nYSejocQuIUvi", "VF0Tok-dHlA17rbZQ0lhmoK_onXsUt0qJM5dlSXs")

def check_for_crew(photo):
    global isCrew
    isCrew = False

    global known_face_encodings
    global firstRun
    known_face_file_path = []
    known_face_names = []
    photo_encoding = []

    try:
        photo_encoding = face_recognition.face_encodings(photo)[0]
    except:
        try:
            photo = face_recognition.load_image_file("./face.jpg")
            photo_encoding = face_recognition.face_encodings(photo)[0]
        except:
            try:
                photo = face_recognition.load_image_file("./face.jpg")
                photo_encoding = face_recognition.face_encodings(photo)[0]
            except:
                print("BAD FRAME")
                return
    cwd = os.getcwd()
    print("SCANNING CREW NAMES")
    for file in os.listdir(cwd + "/crew"):
        if file.endswith(".jpg"):
            nameCrew = str(os.path.join(file)[:-4])
            known_face_names.append(str(nameCrew.replace("_"," ")))

    # SCAN FOLDER FOR CREW
    # only during the first run
    if firstRun == True:
        cwd = os.getcwd()
        for file in os.listdir(cwd + "/crew"):
            print("SETTING UP ENCODING")
            if file.endswith(".jpg"):
                nameCrew = str(os.path.join(file)[:-4])
                known_face_encodings.append(str(nameCrew + "_face_encoding"))
                known_face_file_path.append(str("./crew/" + nameCrew + ".jpg"))

    # Load crew pictures and learn how to recognize it: ENCODING
    # only during the first run
    if firstRun == True:
        x = 0
        for encoding in known_face_encodings:
            print("ENCODING:")
            print(encoding)
            print(known_face_file_path[x])
            face_image = face_recognition.load_image_file(known_face_file_path[x])
            known_face_encodings[x] = face_recognition.face_encodings(face_image)[0]
            x = x + 1
        firstRun = False

    print("SEARCHING FOR CREW")
    matches = face_recognition.compare_faces(known_face_encodings, photo_encoding)

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
        print("FOUND CREW: " + str(name))
        isCrew = True
    else:
        print("NO CREW FOUND")

def get_demographics(face):
    model = app.models.get('demographics')
    response = model.predict([face])
    jsonRes = json.dumps(response)
    jsonRes = json.loads(jsonRes)
    # print(jsonRes['outputs'][0]['data'])
    return jsonRes

def face_detector(new_image):
    image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # rotate image 90 degrees
    rows,cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
    dst = cv2.warpAffine(image,M,(cols,rows))

    # detect faces
    face_classifier = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(dst, 1.3, 4)

    now = datetime.datetime.now()
    nowStrf = now.strftime("%Y%m%d%H%M")
    global serial
    global faceDetected
    global previous_image
    global person
    global payload
    global eye
    global isCrew

    if faces is ():
        sys.stdout.write('.')
        sys.stdout.flush()

    for (x,y,w,h) in faces :
        x = x - 50
        y = y - 50
        w = w + 50
        h = h + 75
        face_image = dst[y : y + h, x : x + w]
        if (not new_image is None and not previous_image is None and not face_image is None):
            # detect eyes
            eye_classifier = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
            eyes = face_classifier.detectMultiScale(face_image, 1.3, 4)
            for (ex, ey, ew, eh) in eyes :
                eye = eye + 1

            if eye < 2 :
                eye = eye
            else:
                # RESIZE IMAGE
                try:
                    image_write = cv2.resize(face_image, (400,400), 0,0,interpolation=cv2.INTER_CUBIC)
                except:
                    try:
                        image_write = cv2.resize(face_image, (400,400), 0,0,interpolation=cv2.INTER_CUBIC)
                    except:
                        try:
                            image_write = cv2.resize(face_image, (400,400), 0,0,interpolation=cv2.INTER_CUBIC)
                        except:
                            return
                else:
                    # CHECK FOR CREW
                    cv2.imwrite("./face.jpg", image_write)
                    crewFrame = cv2.cvtColor(image_write, cv2.COLOR_GRAY2RGB)
                    check_for_crew(crewFrame)

                    if isCrew == True:
                        print("")
                    else:
                        # PREP FOR SAMES PERSON CHECK
                        face = ClImage(filename='./face.jpg')
                        # cv2.imshow("face", image_write)

                        # face_match(new_image, previous_image)
                        surf = cv2.xfeatures2d.SURF_create(400, 5, 5)
                        keypoints1, des1 = surf.detectAndCompute(new_image, None)
                        keypoints2, des2 = surf.detectAndCompute(previous_image, None)

                        #Matcher
                        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

                        #Matching
                        matches = bf.match(des1, des2)
                        matches = sorted(matches, key=lambda val: val.distance)

                        #Result
                        print("MATCHES: " + str(len(matches)))

                        if len(matches) < 500:
                            person = person + 1
                        else:
                            person = person

                        personNumber = "person" + str(person)
                        try:
                            cResult = get_demographics(face)
                        except:
                            print("ClarifaiApp GET ERROR")
                            print(cResult)
                        else:
                            # print(cResult)
                            try:
                                origin = cResult['outputs'][0]['data']['regions'][0]['data']['face']['multicultural_appearance']['concepts'][0]['name']
                                age = cResult['outputs'][0]['data']['regions'][0]['data']['face']['age_appearance']['concepts'][0]['name']
                                sex = cResult['outputs'][0]['data']['regions'][0]['data']['face']['gender_appearance']['concepts'][0]['name']
                                previous_image = new_image
                            except:
                                print("ClarifaiApp FACE DETECTION ERROR")
                                if person == 0 :
                                    person = 0
                                else:
                                    person = person - 1
                                origin = "ClarifaiAppFaceDetectionERROR"
                                age = "ClarifaiAppFaceDetectionERROR"
                                sex = "ClarifaiAppFaceDetectionERROR"
                                print(cResult)
                            else:
                                # print(cResult['outputs'][0]['data']['regions'][0]['data']['face'])
                                print("appearance: " + origin)
                                print("age: " + age)
                                print("sex: " + sex)
                        print(nowStrf + str(serial), "FACE DETECTED !!" + " person" + str(person))

                            #SAVE IMAGE
                        cv2.imwrite("./faces/FaceDetected" + nowStrf + "-" + str(serial) + "-person" + str(person) + ".jpg", image_write)

                            # SEND TO API
                        # payload = {"RCSlocation": "testApp", "RCSmatches": str(len(matches)), "RCSstamp": str(nowStrf + "-" + str(serial) + "-person" + str(person)), "RCSperson": personNumber, "ClarifaiAppOrigin": str(origin), "ClarifaiAppAge": str(age), "ClarifaiAppSex": str(sex)}
                        # r = requests.post("http://143.177.59.207:8282/rcs-data/", data=payload)

                        isCrew = False
                        serial = serial + 1

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (1024, 768)
camera.framerate = 4
rawCapture = PiRGBArray(camera, size=(1024, 768))

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    frameArray = frame.array
    if frameArray is not None:
        face_detector(frameArray)

    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
