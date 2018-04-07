# USAGE
# python pi_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

#import necessary packages
import cv2
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import time

# Setting up the Pi GPIO ports 
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)

# Initializing variables for various PINs on the Pi
GPIO_TRIGGER1 = 29      #Left ultrasonic sensor
GPIO_ECHO1 = 31

GPIO_TRIGGER2 = 36      #Front ultrasonic sensor
GPIO_ECHO2 = 37

GPIO_TRIGGER3 = 33      #Right ultrasonic sensor
GPIO_ECHO3 = 35

MOTOR1B=18  #Left Motor
MOTOR1E=22

MOTOR2B=21  #Right Motor
MOTOR2E=19

# Set the Ultrasonic Sensors Triggers as OUTPUT and Echoes as INPUT
GPIO.setup(GPIO_TRIGGER1,GPIO.OUT)  # Trigger
GPIO.setup(GPIO_ECHO1,GPIO.IN)      # Echo
GPIO.setup(GPIO_TRIGGER2,GPIO.OUT)  # Trigger
GPIO.setup(GPIO_ECHO2,GPIO.IN)
GPIO.setup(GPIO_TRIGGER3,GPIO.OUT)  # Trigger
GPIO.setup(GPIO_ECHO3,GPIO.IN)
GPIO.setup(LED_PIN,GPIO.OUT)

# Initialise the Ultrasonic sensors to be off at the start
GPIO.output(GPIO_TRIGGER1, False)
GPIO.output(GPIO_TRIGGER2, False)
GPIO.output(GPIO_TRIGGER3, False)

# Set the Left and Right Motors for forward and reverse mode
GPIO.setup(MOTOR1B, GPIO.OUT)
GPIO.setup(MOTOR1E, GPIO.OUT)

GPIO.setup(MOTOR2B, GPIO.OUT)
GPIO.setup(MOTOR2E, GPIO.OUT)

# Movement Control
def forward():
      GPIO.output(MOTOR1B, GPIO.HIGH)
      GPIO.output(MOTOR1E, GPIO.LOW)
      GPIO.output(MOTOR2B, GPIO.HIGH)
      GPIO.output(MOTOR2E, GPIO.LOW)
     
def reverse():
      GPIO.output(MOTOR1B, GPIO.LOW)
      GPIO.output(MOTOR1E, GPIO.HIGH)
      GPIO.output(MOTOR2B, GPIO.LOW)
      GPIO.output(MOTOR2E, GPIO.HIGH)
     
def rightturn():
      GPIO.output(MOTOR1B,GPIO.LOW)
      GPIO.output(MOTOR1E,GPIO.HIGH)
      GPIO.output(MOTOR2B,GPIO.HIGH)
      GPIO.output(MOTOR2E,GPIO.LOW)
     
def leftturn():
      GPIO.output(MOTOR1B,GPIO.HIGH)
      GPIO.output(MOTOR1E,GPIO.LOW)
      GPIO.output(MOTOR2B,GPIO.LOW)
      GPIO.output(MOTOR2E,GPIO.HIGH)

def stop():
      GPIO.output(MOTOR1E,GPIO.LOW)
      GPIO.output(MOTOR1B,GPIO.LOW)
      GPIO.output(MOTOR2E,GPIO.LOW)
      GPIO.output(MOTOR2B,GPIO.LOW)


def classify_frame(net, inputQueue, outputQueue):
	# keep looping
	while True:
		# check to see if there is a frame in our input queue
		if not inputQueue.empty():
			# grab the frame from the input queue, resize it, and
			# construct a blob from it
			frame = inputQueue.get()
			frame = cv2.resize(frame, (300, 300))
			blob = cv2.dnn.blobFromImage(frame, 0.007843,
				(300, 300), 127.5)

			# set the blob as input to our deep learning object
			# detector and obtain the detections
			net.setInput(blob)
			detections = net.forward()

			# write the detections to the output queue
			outputQueue.put(detections)      
      
# Calculate distance from nearest obstacle
def sonar(GPIO_TRIGGER,GPIO_ECHO):
      start=0
      stop=0
      # Set pins as output and input
      GPIO.setup(GPIO_TRIGGER,GPIO.OUT)  # Trigger
      GPIO.setup(GPIO_ECHO,GPIO.IN)      # Echo
     
      # Set trigger to False (Low)
      GPIO.output(GPIO_TRIGGER, False)
     
      # Allow module to settle
      time.sleep(0.01)
           
      #while distance > 5:
      #Send 10us pulse to trigger
      GPIO.output(GPIO_TRIGGER, True)
      time.sleep(0.00001)
      GPIO.output(GPIO_TRIGGER, False)
      begin = time.time()
      while GPIO.input(GPIO_ECHO)==0 and time.time()<begin+0.05:
            start = time.time()
     
      while GPIO.input(GPIO_ECHO)==1 and time.time()<begin+0.1:
            stop = time.time()
     
      # Calculate pulse length
      elapsed = stop-start
      # Distance pulse travelled in that time is time
      # multiplied by the speed of sound (cm/s)
      distance = elapsed * 34000
     
      # That was the distance there and back so halve the value
      distance = distance / 2
     
      print "Distance : %.1f" % distance
      # Reset GPIO settings
      return distance
      
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the input queue (frames), output queue (detections),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(net, inputQueue,
	outputQueue,))
p.daemon = True
p.start()

# Initilise the Camera
cap = cv2.VideoCapture(1)
cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv.CV_CAP_PROP_FPS, 16)
rawCapture = np.asarray(cap)
# allow the camera to warmup
time.sleep(0.01)

print("[INFO] Resolution of frame is ",cap.get(cv2.CAP_PROP_FRAME_WIDTH),"X",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("[INFO] Initiating capture of frames...")
time.sleep(2.0)
timerate=[]

# Continue to follow person without stopping
while True :
      ret,img = cap.read()
      frame = np.asarray(img)
      frame=cv2.flip(frame,1)
      global centre_x
      global centre_y
      centre_x=0.
      centre_y=0.
      #distance coming from front ultrasonic sensor
      distanceC = sonar(GPIO_TRIGGER2,GPIO_ECHO2)
      #distance coming from right ultrasonic sensor
      distanceR = sonar(GPIO_TRIGGER3,GPIO_ECHO3)
      #distance coming from left ultrasonic sensor
      distanceL = sonar(GPIO_TRIGGER1,GPIO_ECHO1)
      print("Center : ",distanceC)
      print("Right : ",distanceR)
      print("Left : ",distanceL)
      
      if inputQueue.empty():
            inputQueue.put(frame)

	# if the output queue *is not* empty, grab the detections
      if not outputQueue.empty():
            detections = outputQueue.get()
            
      area=0
       if detections is not None:
		# loop over the detections
            for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
                  confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence`
			# is greater than the minimum confidence
                  if confidence < args["confidence"]:
                   continue

			# otherwise, extract the index of the class label from
			# the `detections`, then compute the (x, y)-coordinates
			# of the bounding box for the object
                  idx = int(detections[0, 0, i, 1])
                  dims = np.array([fW, fH, fW, fH])
                  box = detections[0, 0, i, 3:7] * dims(startX, startY, endX, endY) = box.astype("int")
            
			# draw the prediction on the frame
                  label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
                  cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
                  cv2.rectangle(frame, (78,58), (82,62),(255,255,255), 2)
                  y = startY - 15 if startY - 15 > 15 else startY + 15
                  cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                                               
            
                  if ((endX-startX)*(endY-startY))>area and CLASSES[idx]=="person":
                        area=(endX-startX)*(endY-startY)
                        center_x=(startX+endX)/2
                        center_y=(startY+endY)/2
                        print("[INFO] Area=",area)   
                  elif((endX-startX)*(endY-startY)<area and CLASSES[idx]=="person"):
                        print("Smaller person detected!") 
                        continue                               #Comment this
                  if(CLASSES[idx]=="person"): 
                        abc="Center of object:",center_x,center_y
                        cv2.putText(frame,str(abc), (100,150),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                        if(distanceC<20):
                              #if ball is too far but it detects something in front of it,then it avoid it and reaches the ball.
                              if distanceR>=15:
                                    print("Right")#rightturn()
                                    time.sleep(0.00625)
                                    stop()
                                    time.sleep(0.0125)
                                    print("Forward")#goforward()
                                    time.sleep(0.00625)
                                    stop()
                                    time.sleep(0.0125)
                                    print("Left")#leftturn()
                                    time.sleep(0.00625)
                              elif distanceL>=15:
                                    print("Left")#leftturn()
                                    time.sleep(0.00625)
                                    stop()
                                    time.sleep(0.0125)
                                    print("Forward")#goforward()
                                    time.sleep(0.00625)
                                    stop()
                                    time.sleep(0.0125)
                                    print("Right")#rightturn()
                                    time.sleep(0.00625)
                                    stop()
                                    time.sleep(0.0125)
                              else:
                                    stop()
                                    print("Stop")
                                    time.sleep(0.01)
                        else:
                              #otherwise it move forward
                              print("Forward")#goforward()
                              time.sleep(0.05)
                              #time.sleep(0.5)
                        if center_x<(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/3):
                              print("Align right")#turnright()
                              time.sleep(0.00625)
                              continue
                        if center_x>(2*cap.get(cv2.CAP_PROP_FRAME_WIDTH)/3):
                              print("Align left")#turnleft()
                              time.sleep(0.00625)
                              continue   
      
      cv2.imshow("Output",frame)
      if(cv2.waitKey(1) & 0xff == ord('q')):
            break
GPIO.cleanup()
cv2.estroyAllWindows()
