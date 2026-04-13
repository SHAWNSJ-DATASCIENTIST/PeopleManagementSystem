import streamlit as st
import streamlit.components.v1 as components
import RPi.GPIO as GPIO
import cv2
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from picamera2 import Picamera2
from ultralytics import YOLO
from flask import Flask, Response

IR_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

model = YOLO("yolov8n_ncnn_model")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 640)}))
picam2.start()

shared = {"frame": None, "count": 0}
lock = threading.Lock()

MAX_PEOPLE_RISK = 8

def inference_loop():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = model(frame, imgsz=640, verbose=False, classes=[0], conf=0.2)
        annotated = results[0].plot()
        count = len(results[0].boxes)
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with lock:
            shared["frame"] = jpeg.tobytes()
            shared["count"] = count

threading.Thread(target=inference_loop, daemon=True).start()

flask_app = Flask(__name__)

@flask_app.route("/video")
def video():
    def generate():
        while True:
            with lock:
                frame = shared["frame"]
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

threading.Thread(target=lambda: flask_app.run(host="0.0.0.0", port=5000), daemon=True).start()


def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if b != a else 0.0
    return (c - x) / (c - b) if c != b else 0.0


def fuzz_evaluate(density, chaos):
    d_low  = triangular(density, -0.01, 0.0, 0.5)
    d_med  = triangular(density,  0.25, 0.5, 0.75)
    d_high = triangular(density,  0.5,  1.0, 1.01)
    c_low  = triangular(chaos,   -0.01, 0.0, 0.5)
    c_med  = triangular(chaos,    0.25, 0.5, 0.75)
    c_high = triangular(chaos,    0.5,  1.0, 1.01)

    r_high = max(d_high, c_high)
    r_med  = min(d_med,  c_med)
    r_low  = max(d_low, c_low, min(d_med, c_low), min(c_med, d_low))

    xs = np.linspace(0, 100, 201)
    def _out_low(x):  return triangular(x/100, 0.0,  0.0, 0.5)
    def _out_med(x):  return triangular(x/100, 0.25, 0.5, 0.75)
    def _out_high(x): return triangular(x/100, 0.5,  1.0, 1.0)

    agg = np.maximum.reduce([
        np.minimum(r_low,  [_out_low(x)  for x in xs]),
        np.minimum(r_med,  [_out_med(x)  for x in xs]),
        np.minimum(r_high, [_out_high(x) for x in xs]),
    ])
    centroid = (agg * xs).sum() / agg.sum() if agg.sum() > 0 else 0.0
    crisp = float(centroid / 100.0)

    return {
        "density_memberships": {"low": d_low, "med": d_med, "high": d_high},
        "chaos_memberships":   {"low": c_low, "med": c_med, "high": c_high},
        "rule_outputs":        {"low": r_low, "med": r_med, "high": r_high},
        "crisp": crisp,
    }


def fuse(c_risk, c_conf, p_risk):
    if c_conf > 0.5:
        w_cam, w_sensor = 0.6, 0.4
    else:
        w_cam, w_sensor = 0.1, 0.9
    total = c_risk * w_cam + p_risk * w_sensor
    if total > 0.8:
        status = "CRITICAL"
    elif total > 0.5:
        status = "WARNING"
    else:
        status = "SAFE"
    return {"total_risk": round(total, 3), "status": status,
            "weights": {"cam": w_cam, "piezo": w_sensor}}


if "risk_history" not in st.session_state:
    st.session_state["risk_history"] = []


st.set_page_config(page_title="CrowdSense AI", layout="wide")

with st.sidebar:
    st.title("Settings")
    auto_refresh = st.toggle("Live Auto-Refresh", value=True)
    st.markdown("**Risk Thresholds**")
    warn_thresh = st.slider("Warning threshold", 0.0, 1.0, 0.5, 0.05)
    crit_thresh = st.slider("Critical threshold", 0.0, 1.0, 0.8, 0.05)
    st.divider()
    st.caption("CrowdSense AI — Raspberry Pi Edition")

tab_dash, tab_fuzzy = st.tabs(["Dashboard", "Fuzzy Engine"])

with tab_dash:
    st.title("Crowd Safety Dashboard")

    with lock:
        people_count = shared["count"]

    ir_value = 0 if GPIO.input(IR_PIN) == 0 else 1

    cam_risk  = min(1.0, people_count / MAX_PEOPLE_RISK)
    cam_conf  = 1.0
    p_risk    = float(ir_value)
    fusion    = fuse(cam_risk, cam_conf, p_risk)

    hist = st.session_state["risk_history"]
    hist.append(fusion["total_risk"])
    st.session_state["risk_history"] = hist[-60:]

    status = fusion["status"]
    if status == "CRITICAL":
        st.error(f"{status} — Total Risk: {fusion['total_risk']:.2f}")
    elif status == "WARNING":
        st.warning(f"{status} — Total Risk: {fusion['total_risk']:.2f}")
    else:
        st.success(f"{status} — Total Risk: {fusion['total_risk']:.2f}")

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("People Detected", people_count)
    m2.metric("Camera Risk",     f"{cam_risk:.2f}")
    m3.metric("IR Sensor",       ir_value)
    m4.metric("Total Risk",      f"{fusion['total_risk']:.2f}")

    st.divider()

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Live Camera")
        components.html('<img src="http://localhost:5000/video" width="100%">', height=500)

    with col_right:
        st.subheader("Risk Analysis")

        st.markdown("**Camera Risk**")
        st.progress(cam_risk)

        st.markdown("**IR Sensor Risk**")
        st.progress(p_risk)

        st.markdown("**Total Fused Risk**")
        st.progress(fusion["total_risk"])

        w = fusion["weights"]
        st.caption(f"Weights — Cam: {w['cam']} | Sensor: {w['piezo']}")

        if len(st.session_state["risk_history"]) > 1:
            st.subheader("Total Risk — Last 60s")
            st.line_chart(pd.DataFrame({"Risk": st.session_state["risk_history"]}), height=180)


with tab_fuzzy:
    st.title("Adaptive Neuro-Fuzzy Engine")

    with lock:
        live_count = shared["count"]
    live_ir = 0 if GPIO.input(IR_PIN) == 0 else 1

    live_density = min(1.0, live_count / MAX_PEOPLE_RISK)
    live_chaos   = float(live_ir)

    live_sync = st.toggle("Live Sync — use real-time sensor values", value=True)

    col_ctrl, col_out = st.columns([1, 1.4])

    with col_ctrl:
        if live_sync:
            st.markdown("**Live Sensor Inputs**")
            st.info(f"Crowd Density: **{live_density:.3f}**\nIR Chaos: **{live_chaos:.3f}**")
            density_in = live_density
            chaos_in   = live_chaos
        else:
            st.markdown("**Manual Inputs**")
            density_in = st.slider("Crowd Density", 0.0, 1.0, 0.5, 0.01)
            chaos_in   = st.slider("Chaos Score",   0.0, 1.0, 0.3, 0.01)

    result = fuzz_evaluate(density_in, chaos_in)
    crisp  = result["crisp"]

    with col_out:
        mode_label = "LIVE" if live_sync else "MANUAL"
        if crisp > 0.8:
            st.error(f"[{mode_label}] Crisp Risk: **{crisp:.3f}** — CRITICAL")
        elif crisp > 0.5:
            st.warning(f"[{mode_label}] Crisp Risk: **{crisp:.3f}** — WARNING")
        else:
            st.success(f"[{mode_label}] Crisp Risk: **{crisp:.3f}** — SAFE")

        st.progress(crisp)
        st.caption(f"Density: {density_in:.3f} | Chaos: {chaos_in:.3f} | Risk: {crisp:.3f}")

    st.divider()
    st.subheader("Membership Functions Visualization")

    xs = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=100)
    fig.patch.set_facecolor("#0e1117")

    for ax, val, label in [
        (axes[0], density_in, f"Crowd Density Input  [current = {density_in:.3f}]"),
        (axes[1], chaos_in,   f"Chaos Score Input  [current = {chaos_in:.3f}]"),
    ]:
        ax.set_facecolor("#1a1a2e")
        params = [(0, 0, 0.5), (0.25, 0.5, 0.75), (0.5, 1, 1.01)]
        colors = ["#00e0ff", "#ffff00", "#ff3366"]
        labs   = ["Low", "Medium", "High"]

        for (a, b, c), col, lab in zip(params, colors, labs):
            ys = [triangular(x, a, b, c) for x in xs]
            ax.plot(xs, ys, color=col, linewidth=3.5, label=lab, alpha=1.0)
            ax.fill_between(xs, ys, alpha=0.2, color=col)

        ax.axvline(val, color="#00ff88", linestyle="--", linewidth=3,
                   label=f"Input = {val:.3f}", zorder=10)
        dm_low  = triangular(val, 0,    0,   0.5)
        dm_med  = triangular(val, 0.25, 0.5, 0.75)
        dm_high = triangular(val, 0.5,  1.0, 1.01)
        ax.scatter([val, val, val], [dm_low, dm_med, dm_high],
                   color=["#00e0ff", "#ffff00", "#ff3366"],
                   s=120, zorder=11, edgecolor="white", linewidth=1.5)

        ax.set_xlabel("Input Value (0 to 1)", color="white", fontsize=11, fontweight="bold")
        ax.set_ylabel("Membership Degree",    color="white", fontsize=11, fontweight="bold")
        ax.set_title(label, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim([0, 1.1])
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.25, color="white", linestyle=":")
        ax.tick_params(colors="white", labelsize=9)
        ax.legend(fontsize=9, labelcolor="white",
                  facecolor="#1a1a2e", edgecolor="#444", loc="upper right", framealpha=0.95)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
            spine.set_linewidth(1.2)

    plt.tight_layout(pad=2.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.markdown("#### Output Membership Functions (Risk)")
    fig, ax = plt.subplots(figsize=(14, 5), dpi=100)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0e1117")

    xs_out        = np.linspace(0, 1, 300)
    output_params = [(0.0, 0.0, 0.5), (0.25, 0.5, 0.75), (0.5, 1.0, 1.0)]
    colors_out    = ["#00e0ff", "#ffff00", "#ff3366"]
    labs_out      = ["Low Risk", "Medium Risk", "High Risk"]
    rule_strengths = result["rule_outputs"]

    for (a, b, c), col, lab, sk in zip(output_params, colors_out, labs_out, ["low", "med", "high"]):
        ys = [triangular(x, a, b, c) for x in xs_out]
        ax.plot(xs_out, ys, color=col, linewidth=3.5,
                label=f"{lab}  (a = {rule_strengths[sk]:.3f})", alpha=1.0)
        clipped = [min(y, rule_strengths[sk]) for y in ys]
        ax.fill_between(xs_out, clipped, alpha=0.35, color=col)

    ax.axvline(crisp, color="#00ff88", linestyle="-", linewidth=3,
               label=f"Crisp Output = {crisp:.3f}", zorder=5)
    ax.scatter([crisp], [0.02], color="#00ff88", s=500, marker="^",
               zorder=6, edgecolor="white", linewidth=2)

    ax.set_xlabel("Risk Output (0 to 1)", color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("Membership Degree",    color="white", fontsize=11, fontweight="bold")
    ax.set_title(f"Output Membership (Defuzzified = {crisp:.3f})",
                 color="white", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim([0, 1.1])
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.25, color="white", linestyle=":")
    ax.tick_params(colors="white", labelsize=9)
    ax.legend(fontsize=10, labelcolor="white",
              facecolor="#1a1a2e", edgecolor="#555", loc="upper right", framealpha=0.95)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
        spine.set_linewidth(1.2)

    plt.tight_layout(pad=2.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    st.markdown("#### Membership Degree Values")
    col_d, col_c = st.columns(2)

    def mf_table(memberships, title):
        df = pd.DataFrame({"Membership": {k.capitalize(): round(v, 4) for k, v in memberships.items()}})
        st.markdown(f"**{title}**")
        st.dataframe(df, use_container_width=True)

    with col_d:
        mf_table(result["density_memberships"], f"Density Memberships [input={density_in:.3f}]")
    with col_c:
        mf_table(result["chaos_memberships"],   f"Chaos Memberships [input={chaos_in:.3f}]")

    st.markdown("---")
    st.subheader("Rule Fire Strengths")
    r = result["rule_outputs"]
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Low Risk",    f"{r['low']:.4f}")
    rc2.metric("Medium Risk", f"{r['med']:.4f}")
    rc3.metric("High Risk",   f"{r['high']:.4f}")

    st.divider()
    st.subheader("Live Sensor Readings")
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Camera Density Input", f"{live_density:.3f}")
    lc2.metric("IR Chaos Input",       f"{live_chaos:.3f}")
    live_result = fuzz_evaluate(live_density, live_chaos)
    lc3.metric("Live Fuzzy Risk Out",  f"{live_result['crisp']:.3f}")


if auto_refresh or live_sync:
    time.sleep(1.5)
    st.rerun()