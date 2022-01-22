#people_detection.py

from imageai.Detection import ObjectDetection
import os
import logging
logging.getLogger('tensorflow').disabled = True


person_count = 0
car_count = 0
other_object_count = 0
no_of_images = 7   #(image0.jpg to image6.jpg)

print("Running Senzopt People Detection and Counting Algorithm.....")
print("Total no of images to be processed is:",no_of_images)
print("Getting the Anchors setup for images.....")

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

print("Calculating detection probabilities......")
print("Almost done....")
print("Getting Results...")
print(" ")


for i in range(0,no_of_images):
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image"+str(i)+".jpg"), output_image_path=os.path.join(execution_path , "imagenew"+str(i)+".jpg"))
    for eachObject in detections:
        if eachObject["name"] == 'person':
            person_count+=1
            #print(eachObject["name"] ,person_count, " : " , eachObject["percentage_probability"] )
        elif eachObject["name"] == 'car':
            car_count+=1
            #print(eachObject["name"] ,car_count, " : " , eachObject["percentage_probability"] )
        else:
            other_object_count+=1            
            pass

    print("Total person detection count for image" +str(i)+ " is :", person_count)
    print("Total car detection count for image" +str(i)+ " is :", car_count)
    print("other objects detection count for image" +str(i)+ " is :", other_object_count)
    print(" ")
    person_count = 0
    car_count = 0
    other_object_count = 0
