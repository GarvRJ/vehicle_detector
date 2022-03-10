import cv2
import time
from _datetime import datetime
import collections
import numpy as np
import requests
from firebase_admin import credentials, firestore
import pyrebase
import firebase_admin

station = "Station B"
station_id = "5"
loc = {
    "Latitude": "19.044197669339596",
    "Longitude": "72.86488164388184"
}
services = ["Petrol", "Speed", "Air Filling", "Diesel", "CNG"]
cred = credentials.Certificate('./ServiceAccount.json')
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()
doc_ref = db.collection(u'Pumps').document(u'{}'.format(station))
doc_ref.set({
    u'Services': services,
    u'Location': loc
})
config = {
    "apiKey": "AIzaSyDef18Xg_Ri5Kvnc8VYabur4yVMdVvOAsw",
    "authDomain": "miniproject-442bd.firebaseapp.com",
    "projectId": "miniproject-442bd",
    "databaseURL": "https://miniproject-442bd-default-rtdb.asia-southeast1.firebasedatabase.app",
    "storageBucket": "miniproject-442bd.appspot.com",
    "messagingSenderId": "115419548616",
    "appId": "1:115419548616:web:18c046d397d5bfefcf8efd",
    "measurementId": "G-LCWK79KJ7C"
}
firebase = pyrebase.initialize_app(config)
database = firebase.database()
database.update({station: "Online"})
print('{} with {} at {}'.format(station, services, loc))
webcam = 0
cctv = 'rtsp://192.168.0.169/live/ch00_1'
n = 'night.mp4'
p = 'pump.mp4'
sd = 'sample_drive.mp4'
vd = 'video.mp4'

def build_model(is_cuda):
    net = cv2.dnn.readNet("yolov5s.onnx")
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4
cs = collections.deque(maxlen=35)
bs = collections.deque(maxlen=35)
count = 0


def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds


def load_capture():
    capture = cv2.VideoCapture(vd)
    return capture


def load_classes():
    class_list = []
    with open("classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list


class_list = load_classes()


def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

is_cuda = True

net = build_model(is_cuda)
capture = load_capture()

start = time.time_ns()
frame_count = 0
total_frames = 0
fps = -1


def log(sid, bks, crs):
    url = "http://pawnest.com/webservice/log.php"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
    }
    dt = {'station_id': sid, 'bikes': bks, 'cars': crs}
    x = requests.post(url, data=dt, headers=headers)
    print(x.text)


while True:

    _, frame = capture.read()
    if frame is None:
        print("End of stream")
        break
    car_count = 0
    bike_count = 0
    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)

    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

    frame_count += 1
    total_frames += 1

    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        if classid == 2:
            car_count += 1
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
        elif classid == 3:
            bike_count += 1
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
        cs.append(car_count)
        bs.append(bike_count)
        count += 1
        # print(count)
        if count % 35 == 0:
            # print(str(max(cs)))
            print('UPDATING DATA ' + str(int(count / 35)) + '\n CARS:\t' + str(max(cs)) + '\n BIKES:\t' + str(max(bs)))
            countUp = {"cars": max(cs), "bikes": max(bs)}
            upload = {
                station: countUp
            }
            database.update(upload)
            now = datetime.now()
            doc_upd = db.collection(u'Updates').document(u'{}'.format(now))
            doc_upd.set(
                {
                    u'{}'.format(station): {
                        u'Bikes': max(bs),
                        u'Cars': max(cs)
                    }
                }
            )
            log(station_id, max(bs), max(cs))
    if frame_count >= 30:
        end = time.time_ns()
        fps = 1000000000 * frame_count / (end - start)
        frame_count = 0
        start = time.time_ns()

    if fps > 0:
        fps_label = "FPS: %.2f" % fps
        cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("output", frame)

    if cv2.waitKey(1) > -1:
        print("finished by user")
        break

print("Total frames: " + str(total_frames))
database.update({station: "Offline"})
capture.release()
cv2.destroyAllWindows()
