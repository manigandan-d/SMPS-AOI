# =========================================================
# AOI + CONVEYOR SYSTEM (FINAL FIXED FSM)
# =========================================================

from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import time
import numpy as np
import RPi.GPIO as GPIO

# =========================================================
# GPIO SETUP
# =========================================================
GPIO.setmode(GPIO.BCM)

AIN1, AIN2, PWMA, STBY = 17, 27, 22, 23
START_BTN, STOP_BTN = 5, 6

GPIO.setup([AIN1, AIN2, STBY], GPIO.OUT)
GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(START_BTN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(STOP_BTN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

pwm = GPIO.PWM(PWMA, 1000)

def motor_run():
    GPIO.output(AIN1, GPIO.HIGH)
    GPIO.output(AIN2, GPIO.LOW)
    GPIO.output(STBY, GPIO.HIGH)
    pwm.start(70)

def motor_stop():
    pwm.stop()
    GPIO.output(STBY, GPIO.LOW)

# =========================================================
# YOLO MODEL (NCNN)
# =========================================================
model = YOLO("yolo11n_ncnn_model", task="detect")

CLASSES = ["fuse", "hv_capacitor", "mov", "optocoupler"]
EXPECTED = set(CLASSES)

# =========================================================
# CAMERA
# =========================================================
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280,720), "format":"RGB888"}
    )
)
picam2.start()
time.sleep(1)

# =========================================================
# ROI + ENTRY DETECTION (VISION)
# =========================================================
ROI = (400, 200, 900, 550)
ROI_TRIGGER_FRAMES = 5
DIFF_PIXEL_THRESHOLD = 9000

BASELINE_GRAY = None

def pcb_entered_roi(frame):
    global BASELINE_GRAY

    x1,y1,x2,y2 = ROI
    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    if BASELINE_GRAY is None:
        BASELINE_GRAY = gray.copy()
        return False

    diff = cv2.absdiff(BASELINE_GRAY, gray)
    _, th = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    return cv2.countNonZero(th) > DIFF_PIXEL_THRESHOLD

# =========================================================
# FSM VARIABLES
# =========================================================
STATE = "IDLE"
roi_counter = 0

RESULT_HOLD_TIME = 3.0
result_start_time = 0
LAST_RESULT = None
LAST_MISSING = set()

COOLDOWN_TIME = 1.2
cooldown_start_time = 0

# =========================================================
# MAIN LOOP
# =========================================================
try:
    while True:

        # ---------- EMERGENCY STOP ----------
        if GPIO.input(STOP_BTN) == 0:
            motor_stop()
            STATE = "IDLE"
            BASELINE_GRAY = None
            frame = np.zeros((720,1280,3), dtype=np.uint8)
            cv2.putText(frame,"EMERGENCY STOP",(360,360),
                        cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,255),3)
            cv2.imshow("AOI", frame)
            cv2.waitKey(1)
            continue

        frame = picam2.capture_array()
        cv2.rectangle(frame, ROI[:2], ROI[2:], (255,255,0), 2)

        # ---------- IDLE ----------
        if STATE == "IDLE":
            cv2.putText(frame,"SYSTEM IDLE - PRESS START",
                        (300,360),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            if GPIO.input(START_BTN) == 0:
                motor_run()
                BASELINE_GRAY = None
                STATE = "RUNNING"
                time.sleep(0.5)

        # ---------- RUNNING ----------
        elif STATE == "RUNNING":
            cv2.putText(frame,"CONVEYOR RUNNING",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            if pcb_entered_roi(frame):
                roi_counter += 1
                if roi_counter >= ROI_TRIGGER_FRAMES:
                    motor_stop()
                    time.sleep(0.4)
                    roi_counter = 0
                    STATE = "INSPECTING"
            else:
                roi_counter = 0

        # ---------- INSPECTING ----------
        elif STATE == "INSPECTING":
            results = model(frame, imgsz=640, conf=0.4, verbose=False)
            detected = set()

            for r in results:
                if r.boxes is None: continue
                for cls in r.boxes.cls:
                    detected.add(model.names[int(cls)])

            LAST_MISSING = EXPECTED - detected
            LAST_RESULT = "PASS" if not LAST_MISSING else "FAIL"

            result_start_time = time.time()
            STATE = "RESULT_DISPLAY"

        # ---------- RESULT DISPLAY ----------
        elif STATE == "RESULT_DISPLAY":
            bar_color = (0,180,0) if LAST_RESULT=="PASS" else (0,0,255)
            cv2.rectangle(frame,(0,0),(1280,80),bar_color,-1)
            cv2.putText(frame,f"STATUS : {LAST_RESULT}",
                        (20,55),
                        cv2.FONT_HERSHEY_SIMPLEX,1.4,(255,255,255),3)

            y = 110
            for c in CLASSES:
                txt,col = (f"{c} : OK",(0,200,0)) if c not in LAST_MISSING \
                          else (f"{c} : MISSING",(0,0,255))
                cv2.putText(frame,txt,(20,y),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,col,2)
                y += 35

            if time.time() - result_start_time >= RESULT_HOLD_TIME:
                motor_run()
                BASELINE_GRAY = None
                cooldown_start_time = time.time()
                STATE = "COOLDOWN"

        # ---------- COOLDOWN ----------
        elif STATE == "COOLDOWN":
            cv2.putText(frame,"COOLDOWN",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

            if time.time() - cooldown_start_time >= COOLDOWN_TIME:
                BASELINE_GRAY = None
                STATE = "RUNNING"

        cv2.imshow("AOI", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# =========================================================
# CLEANUP
# =========================================================
finally:
    motor_stop()
    picam2.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
