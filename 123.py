import cv2  # pip install opencv-python
import numpy  # pip install numpy
from ctypes import windll
from glob import glob
from PIL import Image as im

def chebupelina(imgn):
    from functools import reduce
    net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg", r"yolov3_custom_4000.weights")
    # classes = []
    with open('classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    my_img = cv2.imread(imgn)  # tut photo
    #my_img = cv2.resize(my_img, (xx, yy), fx=0, fy=0)
    ht, wt, _ = my_img.shape
    blob = cv2.dnn.blobFromImage(my_img, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    last_layer = net.getUnconnectedOutLayersNames()
    layer_out = net.forward(last_layer)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_out:
        for detection in output:
            score = detection[5:]
            class_id = numpy.argmax(score)
            confidence = score[class_id]
            if confidence > .5:
                center_x = int(detection[0] * wt)
                center_y = int(detection[1] * ht)
                w = int(detection[2] * wt)
                h = int(detection[3] * ht)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append([float(confidence)])
                class_ids.append(class_id)
    number_object_detected = len(boxes)
    font = cv2.FONT_HERSHEY_PLAIN
    font = cv2.FONT_HERSHEY_SIMPLEX
    # colors = numpy.random.uniform(0,255,size= (len(boxes),3))
    if not boxes:
        return False
    for i in boxes:
        x, y, w, h = i
        label = str(classes[class_ids[0]])
        confidence = str(numpy.round(confidences[0], 2))
        # print(confidence)
        color = (0, 255, 50)
        cv2.rectangle(my_img, (x, y), (x + w, y + h), color, 3)
        if label == "1":
            cv2.putText(my_img, label + confidence, (x, y + 20), font, 1.5, (0, 0, 0), 2)
        else:
            cv2.putText(my_img, label + confidence, (x, y + 20), font, 1, (0, 0, 0), 2)
    #cv2.imshow('img', my_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    image = im.open(imgn)
    image.save('notstatic/1.jpg')
    image.close()
    cv2.imwrite('notstatic/1.jpg', my_img)
    print("yes")
    return True
# q = glob('withBears/*.jpg')
chebupelina('withBears/_2016-05-12 14-20-10_2145_2R.JPG')
# for i in q:
#     ima = im.open(i)
#     w, h = ima.size
#     fImg = im.new('RGB', (w, h), 'white')
#     for j in range(4):
#         for k in range(4):
#             cropped = ima.crop((j * (w // 4), k * (h // 4), (j + 1) * (w // 4), (k+1) * (w//4)))
#             cropped.save('notstatic/1.jpg')
#             qwewersdf = chebupelina('notstatic/1.jpg')
#             if qwewersdf:
#                 c_1 = im.open('notstatic/2.jpg')
#             else:
#                 c_1 = im.open('notstatic/1.jpg')
#             fImg.paste(c_1, (j * (w //4), k * (h // 4)))
#     fImg.save("wb/wb_" + i.split('\\')[-1])
#     print(1)