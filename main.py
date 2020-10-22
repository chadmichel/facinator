import time
import os
import cv2
import numpy as np

# from: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
face_finder = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

captures_directory = "captures"
people_directory = "people"
test_directory = "tests"

people = []
trained = 0


# other ref https://github.com/informramiz/opencv-face-recognition-python

def setup():
    print("### SETUP ### ")
    if not os.path.exists(captures_directory):
        os.makedirs(captures_directory)
    if not os.path.exists(people_directory):
        os.makedirs(people_directory)

def cleanup():
    print("### CLEANUP ### ")
    for filename in os.listdir(captures_directory):
        print(filename)
        if (filename.endswith(".png")):
            os.remove(captures_directory + "/" + filename)

def train():
    print("### TRAIN ### ")

    faces = []
    labels = []
    labels_int = []
    for person in os.listdir(people_directory):
        personPath = people_directory + "/" + person
        if os.path.isdir(personPath):

            people.append(person)

            for photo in os.listdir(personPath):
                photoPath = personPath + "/" + photo

                if photo != ".DS_Store" and not os.path.isdir(photoPath):

                    image = cv2.imread(photoPath)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    imageFaces = face_finder.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)

                    if len(imageFaces) == 1:
                        (x,y,w,h) = imageFaces[0]
                        faces.append(gray_image[y:y+w, x:x+h])
                        labels.append(person)
                        labels_int.append(len(people) - 1)
                        print("person=" + person + " path=" + photoPath)

    if len(faces) > 0:
        face_recognizer.train(faces, np.array(labels_int))
        face_recognizer.write('trainer.yml')

def test():

    print("### TEST ### ")

    for testDir in os.listdir(test_directory):
        if testDir == ".DS_Store":
            continue
        for photo in os.listdir(test_directory + "/" + testDir):
            if photo != ".DS_Store":
                image = cv2.imread(test_directory + "/" + testDir + "/" + photo)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_finder.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    grayFace = gray[y:y + w, x:x + h]
                    face_label = face_recognizer.predict(grayFace)
                    if len(face_label) > 0:
                        index = face_label[0]
                        if index >= 0 and index < len(people):
                            person = people[face_label[0]]
                            if face_label[1] < 55:
                                # print(photo + ' found person ' + person + ' ' + str(face_label[1]))
                                if person != testDir:
                                    print("TEST ERROR '" + photo + "' '" + person + "' != '" + testDir + "'")
                            else:
                                print(photo + ' LOW QUALITY: ' + person + ' ' + str(face_label[1]))
                        else:
                            print(photo + ' NO PERSON FOUND: ')

def mainLoop():
    cam = cv2.VideoCapture(0)
    print("### CAMERA ### ")

    img_counter = 0

    while len(people) > 0:
        ret, frame = cam.read()

        img_counter += 1
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        if img_counter > 1:
            diff = cv2.compareHist(hist, old_hist, cv2.HISTCMP_CORREL)

            if diff < 0.99:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_finder.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    img_name = "captures/captures_{}.png".format(img_counter)
                    cv2.imwrite(img_name, frame)

                    (x, y, w, h) = faces[0]
                    grayFace = gray[y:y + w, x:x + h]

                    face_label = face_recognizer.predict(grayFace)
                    if len(face_label) > 0:
                        index = face_label[0]
                        if index >= 0 and index < len(people) and face_label[1] < 55:
                            person = people[face_label[0]]
                            print('CAMERA found person ' + person + ' ' + str(face_label[1]))
                else:
                    print("CAMERA NO FACES")

        old_hist = hist

        if img_counter % 1000 == 0:
            cleanup()

        if img_counter > 5000:
            exit(0)

        time.sleep(.25)

    cam.release()
    cv2.destroyAllWindows()


setup()
cleanup()
train()
test()
mainLoop()


