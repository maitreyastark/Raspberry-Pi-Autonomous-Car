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
      
# Initilise the Camera
cap = cv2.VideoCapture(1)
cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv.CV_CAP_PROP_FPS, 16)
rawCapture = np.asarray(cap)
# allow the camera to warmup
time.sleep(0.01)

# Continue to follow person without stopping
while( cap.isOpened() ) :
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
      
      
      
      
      cv2.imshow("Output",frame)
      if(cv2.waitKey(1) & 0xff == ord('q')):
            break
GPIO.cleanup()
