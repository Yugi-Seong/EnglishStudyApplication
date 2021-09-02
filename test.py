import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def yolo(image_path):
    
    # %matplotlib inline
    # To know the shape of images
    def image_and_shapes(image):
        img= plt.imread(image)
        plt.imshow(img)
        print("Shape of the image:{}".format(img.shape))



    #Load YOLO
    net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
    classes = []
    with open("coco.names","r") as f:
        classes = [line.strip() for line in f.readlines()]

    net.getLayerNames()
    net.getUnconnectedOutLayers()
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    colors= np.random.uniform(0,255,size=(len(classes),3))

    #loading image
    img = cv2.imread(image_path)
    img = cv2.resize(img,None,fx=0.4,fy=0.3)
    height,width,channels = img.shape


    image_and_shapes(image_path)

    #detecting objects
    blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)

    net.setInput(blob)
    outs = net.forward(outputlayers)

    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            
            if confidence > 0.5:
                #onject detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
            
                #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                
                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected

    #cv2.dnn.NMSBoxes(boxes,confidences,score_threshold,IOU_threshold)
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)#Non Max Suppressions

    '''Now using below loop over all found boxes, 
    if box is appearing in indexes then only draw rectangle, color it,
    put text of class name on it.'''
    font = cv2.FONT_HERSHEY_PLAIN

    list_label = []
    
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]]) # 출력되는 객체 
            # label을 list 로 만들어서 중복없이 저장 
            list_label.append(label)
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,label,(x,y+30),font,1,(255,255,255),2)
    print(list(set(list_label)))
    list_label = list(set(list_label))


    #cv2.imshow("Image",img)
    #cv2.waitKey(0)
    cv2.imwrite('Image.jpg',img)
    cv2.imwrite("static/img/Image.jpg",img)
    #cv2.destroyAllWindows()

    # return image_and_shapes("Image.jpg")
    return list_label



