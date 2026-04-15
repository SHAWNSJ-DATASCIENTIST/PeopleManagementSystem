import atexit
import time
import threading
from collections import deque

import cv2
import pandas as pd
import streamlit as st
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO

st.set_page_config(page_title="CrowdSense AI", layout="wide")

st.markdown(
    """
    <style>
        .title {
            font-size: 44px;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }
        .subtitle {
            color: #666;
            font-size: 14px;
            margin-bottom: 1rem;
        }
        .banner-safe {
            background: #eaf8ee;
            border: 1px solid #b9e2c4;
            border-radius: 16px;
            padding: 16px 20px;
            font-size: 24px;
            font-weight: 800;
            color: #167a36;
        }
        .banner-warning {
            background: #fff6d8;
            border: 1px solid #f0dc96;
            border-radius: 16px;
            padding: 16px 20px;
            font-size: 24px;
            font-weight: 800;
            color: #8a5b00;
        }
        .banner-critical {
            background: #ffe1e1;
            border: 1px solid #f0a0a0;
            border-radius: 16px;
            padding: 16px 20px;
            font-size: 24px;
            font-weight: 800;
            color: #a31212;
        }
        .card {
            background: #ffffff;
            border: 1px solid #ececec;
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 1px 6px rgba(0,0,0,0.03);
        }
        .label {
            font-size: 13px;
            color: #666;
            margin-bottom: 4px;
        }
        .value {
            font-size: 28px;
            font-weight: 800;
            line-height: 1.1;
        }
        .person-text {
            text-align: center;
            font-size: 34px;
            font-weight: 800;
            margin-top: 10px;
            margin-bottom: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>Crowd Safety Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Camera feed fixed on the left, sensor values on the right</div>", unsafe_allow_html=True)

# Pin Configurations
IR1_PIN = 17
IR2_PIN = 18
PIEZO1_PIN = 23
PIEZO2_PIN = 24
TRIG_PIN = 20
ECHO_PIN = 21
SERVO_PIN = 26  

# Constants
MAX_PEOPLE_RISK = 8
CAMERA_RISK_WEIGHT = 3.0
WARNING_THRESHOLD = 0.50
CRITICAL_THRESHOLD = 0.80

# Standard PyTorch YOLOv8 Nano model
MODEL_PATH = "yolov8n.pt"

# GPIO Setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(IR1_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(IR2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(PIEZO1_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(PIEZO2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
GPIO.output(TRIG_PIN, False)

GPIO.setup(SERVO_PIN, GPIO.OUT)
servo_pwm = GPIO.PWM(SERVO_PIN, 50)  
servo_pwm.start(2.5)  

def cleanup():
    try:
        cam = st.session_state.get("picam2")
        if cam is not None:
            cam.stop()
    except Exception:
        pass
    try:
        servo_pwm.stop()  
    except Exception:
        pass
    try:
        GPIO.cleanup()
    except Exception:
        pass

atexit.register(cleanup)

def read_ir(pin):
    return 1 if GPIO.input(pin) == 0 else 0

def read_piezo(pin):
    return 1 if GPIO.input(pin) == 1 else 0

def get_distance_cm(timeout=0.03):
    try:
        GPIO.output(TRIG_PIN, False)
        time.sleep(0.0002)

        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)

        start = time.time()
        while GPIO.input(ECHO_PIN) == 0:
            if time.time() - start > timeout:
                return None

        pulse_start = time.time()

        while GPIO.input(ECHO_PIN) == 1:
            if time.time() - pulse_start > timeout:
                return None

        pulse_end = time.time()
        return round((pulse_end - pulse_start) * 17150, 1)
    except Exception:
        return None

def clamp01(x):
    return max(0.0, min(1.0, x))

def camera_risk(person_count):
    return clamp01((person_count / MAX_PEOPLE_RISK) * CAMERA_RISK_WEIGHT)

def fuse_risk(person_count, ir1, ir2, p1, p2, dist):
    cam = camera_risk(person_count)
    total = cam + 0.12 * (ir1 + ir2) + 0.18 * (p1 + p2)

    if dist is not None and dist < 150:
        total += clamp01((150 - dist) / 150) * 0.25

    total = clamp01(total)

    if total > CRITICAL_THRESHOLD:
        status = "CRITICAL"
    elif total > WARNING_THRESHOLD:
        status = "WARNING"
    else:
        status = "SAFE"

    sensor_part = clamp01(total - cam)
    return total, cam, sensor_part, status

@st.cache_resource
def load_model():
    # Load the standard PyTorch model
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_camera():
    cam = Picamera2()
    # 640x480 resolution for proper YOLO detection
    cam.configure(cam.create_preview_configuration(main={"size": (640, 480)})) 
    cam.start()
    time.sleep(0.5)
    return cam

try:
    model = load_model()
except Exception as e:
    st.error(f"YOLO model load failed: {e}")
    st.stop()

try:
    picam2 = load_camera()
    st.session_state.picam2 = picam2
except Exception as e:
    st.error(f"Camera load failed: {e}")
    st.stop()

def servo_sweep():
    while True:
        servo_pwm.ChangeDutyCycle(2.5)
        time.sleep(0.5)  
        servo_pwm.ChangeDutyCycle(0)
        time.sleep(5.0)
        
        servo_pwm.ChangeDutyCycle(7.5)
        time.sleep(0.5)  
        servo_pwm.ChangeDutyCycle(0)
        time.sleep(5.0)

servo_thread = threading.Thread(target=servo_sweep, daemon=True)
servo_thread.start()

if "risk_history" not in st.session_state:
    st.session_state.risk_history = deque(maxlen=60)

left_col, right_col = st.columns([1.45, 1.0], gap="large")

with left_col:
    st.markdown("### Camera Feed")
    frame_ph = st.empty()
    person_ph = st.empty()
    chart_ph = st.empty()

with right_col:
    status_ph = st.empty()

    metric_cols = st.columns(4)
    people_ph = metric_cols[0].empty()
    camrisk_ph = metric_cols[1].empty()
    sensorrisk_ph = metric_cols[2].empty()
    totalrisk_ph = metric_cols[3].empty()

    st.markdown("### Sensor Readings")
    s1, s2, s3, s4 = st.columns(4)
    ir1_ph = s1.empty()
    ir2_ph = s2.empty()
    piezo1_ph = s3.empty()
    piezo2_ph = s4.empty()

    st.markdown("### Environment")
    e1, e2, e3, e4 = st.columns(4)
    dist_ph = e1.empty()
    vib_ph = e2.empty()
    irmotion_ph = e3.empty()
    pirmotion_ph = e4.empty()

while True:
    try:
        # picam2 outputs RGB by default
        frame = picam2.capture_array()

        # Convert RGB to BGR for YOLO's internal processing
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run YOLO on the BGR frame
        results = model(bgr_frame, verbose=False)
        person_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Only count persons
                    person_count += 1
                    
                    # Draw green bounding box directly onto the original RGB frame for Streamlit display
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ir1 = read_ir(IR1_PIN)
        ir2 = read_ir(IR2_PIN)
        p1 = read_piezo(PIEZO1_PIN)
        p2 = read_piezo(PIEZO2_PIN)
        dist = get_distance_cm()

        total_risk, cam_risk, sensor_risk, status = fuse_risk(
            person_count, ir1, ir2, p1, p2, dist
        )

        st.session_state.risk_history.append(total_risk)

        if status == "CRITICAL":
            status_ph.markdown(
                f"<div class='banner-critical'>🛑 CRITICAL — Total Risk: {total_risk:.2f}</div>",
                unsafe_allow_html=True,
            )
        elif status == "WARNING":
            status_ph.markdown(
                f"<div class='banner-warning'>⚠️ WARNING — Total Risk: {total_risk:.2f}</div>",
                unsafe_allow_html=True,
            )
        else:
            status_ph.markdown(
                f"<div class='banner-safe'>✅ SAFE — Total Risk: {total_risk:.2f}</div>",
                unsafe_allow_html=True,
            )

        people_ph.metric("People", person_count)
        camrisk_ph.metric("Camera", f"{cam_risk:.2f}")
        sensorrisk_ph.metric("Sensors", f"{sensor_risk:.2f}")
        totalrisk_ph.metric("Risk", f"{total_risk:.2f}")

        # Render the RGB frame with the green boxes drawn on it
        frame_ph.image(frame, channels="RGB", use_container_width=True)
        
        person_ph.markdown(
            f"<div class='person-text'>person - {person_count}</div>",
            unsafe_allow_html=True,
        )

        if len(st.session_state.risk_history) > 1:
            chart_ph.line_chart(
                pd.DataFrame({"Risk": list(st.session_state.risk_history)}),
                height=180,
            )
        else:
            chart_ph.empty()

        ir1_ph.markdown(
            f"<div class='card'><div class='label'>IR1</div><div class='value'>{'YES' if ir1 else 'NO'}</div></div>",
            unsafe_allow_html=True,
        )
        ir2_ph.markdown(
            f"<div class='card'><div class='label'>IR2</div><div class='value'>{'YES' if ir2 else 'NO'}</div></div>",
            unsafe_allow_html=True,
        )
        piezo1_ph.markdown(
            f"<div class='card'><div class='label'>Piezo 1</div><div class='value'>{'YES' if p1 else 'NO'}</div></div>",
            unsafe_allow_html=True,
        )
        piezo2_ph.markdown(
            f"<div class='card'><div class='label'>Piezo 2</div><div class='value'>{'YES' if p2 else 'NO'}</div></div>",
            unsafe_allow_html=True,
        )

        dist_ph.markdown(
            f"<div class='card'><div class='label'>Distance (cm)</div><div class='value'>{'N/A' if dist is None else dist}</div></div>",
            unsafe_allow_html=True,
        )
        vib_ph.markdown(
            f"<div class='card'><div class='label'>Vibration</div><div class='value'>{'YES' if (p1 or p2) else 'NO'}</div></div>",
            unsafe_allow_html=True,
        )
        irmotion_ph.markdown(
            f"<div class='card'><div class='label'>IR Motion</div><div class='value'>{'YES' if (ir1 or ir2) else 'NO'}</div></div>",
            unsafe_allow_html=True,
        )
        pirmotion_ph.markdown(
            f"<div class='card'><div class='label'>PIR Motion</div><div class='value'>{'YES' if (p1 or p2) else 'NO'}</div></div>",
            unsafe_allow_html=True,
        )

        time.sleep(0.05)

    except Exception as e:
        st.error(f"Runtime error: {e}")
        time.sleep(0.2)