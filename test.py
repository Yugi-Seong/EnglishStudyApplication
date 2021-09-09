import cv2
import numpy as np
import matplotlib.pyplot as plt

def yolo(image_path):
    
    # %matplotlib inline
    # To know the shape of images
    def image_and_shapes(image):
        img= plt.imread(image)
        plt.imshow(img)
        print("Shape of the image:{}".format(img.shape))

    whT = 320
    confThreshold = 0.5
    nmsThreshold = 0.3
    
    classesFile = 'coco.names'
    classNames = []

    # model 생성 
    modelConfiguration = 'yolov3.cfg'
    modelWeights = 'yolov3.weights'

    #Load YOLO

    with open(classesFile,'rt') as f :
        classNames = f.read().rstrip('\n').split('\n')

    net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    classes = []
    with open("coco.names","r") as f:
        classes = [line.strip() for line in f.readlines()]
        
            #loading image
    img = cv2.imread(image_path)
    img = cv2.resize(img,None,fx=0.4,fy=0.3)
    height,width,channels = img.shape


    image_and_shapes(image_path)


    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    
    colors= np.random.uniform(0,255,size=(len(classes),3))
    
    # 객체 인식 
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
                confs.append(float(confidence))
                # 객체 인식한 coco.names 의 index 
                #print(classId)
                
                # 객체 index를 이용해 사물 출력 
                print("yolo출력확인 : {}".format(classNames[classId]))
                           
    
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)  
    
    list_label = []

    for i in indices :
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        label = str(classes[classIds[i]])
        
        list_label.append(label)
        color = colors[i]
        
        
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        
    
    
    print(list(set(list_label)))
    list_label = list(set(list_label))


    #cv2.waitKey(0)
    cv2.imwrite('Image.jpg',img)
    cv2.imwrite("static/img/Image.jpg",img)
    #cv2.destroyAllWindows()

#     image_and_shapes("Image.jpg")
    return list_label