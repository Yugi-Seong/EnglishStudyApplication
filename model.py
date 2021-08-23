import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = 'coco.names'
classNames = []

# defalt로 text 읽기 
with open(classesFile,'rt') as f :
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold :
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                # 객체 인식한 coco.names 의 index 
                print(classId)
                # 객체 index를 이용해 사물 출력 
                print("model.py출력확인 : {}".format(classNames[classId]))
                object = classNames[classId] 
                confs.append(float(confidence))
       
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)  

    

    #print(indices)
    for i in indices :
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

    return object
    
    

def photo():
    # while True :
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)
    
    layerNames = net.getLayerNames()
    #print(layerNames)
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)
    #print(net.getUnconnectedOutLayers())

    outputs = net.forward(outputNames)
    findObjects(outputs,img)
    TEXT = findObjects(outputs,img)
    
    cv2.imshow('Image',img)
    cv2.imwrite('capture_images/image1.png',img,params=[cv2.IMWRITE_PNG_COMPRESSION,0])
    cv2.waitKey(2)
    #cv2.destroyAllWindows(2)
    # break 

    return TEXT
    