# PeopleManagementSystem

This project implements a hybrid soft computing framework combining deep learning and evolutionary optimization for intelligent crowd risk assessment. The system leverages CNN for spatial density classification, YOLO for object detection, fuzzy inference for uncertainty modeling, and genetic algorithms for rule optimization.

📌 Crowd Management System using Soft Computing Techniques
🔍 Project Overview

This project presents an AI-driven Crowd Management and Safety System built using hybrid Soft Computing techniques, integrating:

Convolutional Neural Networks (CNN)

YOLO (You Only Look Once)

Fuzzy Logic Inference System

Genetic Algorithm Optimization

The system monitors crowd density in real time, evaluates risk levels, and generates adaptive safety alerts.

🧠 Soft Computing Approach

This project is based on the principles of Soft Computing, which deals with uncertainty, approximation, and partial truth.

1️⃣ CNN (Convolutional Neural Network)

Used for crowd density estimation

Classifies images into:

Low density

Medium density

High density

2️⃣ YOLO Object Detection

Real-time person detection

Counts number of people in frame

Used for live surveillance analysis

3️⃣ Fuzzy Logic Controller

Handles uncertainty in:

Crowd density

Vibration level

Movement intensity

Generates risk levels:

Safe

Warning

Critical

4️⃣ Genetic Algorithm

Optimizes fuzzy rule weights

Improves system accuracy

Evolves best threshold parameters

Camera Feed → YOLO → Person Count
↓
CNN Density
↓
Fuzzy Inference System
↓
Risk Classification
↓
Alert / Dashboard Output

🚀 Technologies Used

Python

TensorFlow / Keras

PyTorch (YOLO)

OpenCV

NumPy

SkFuzzy

Genetic Algorithm (custom implementation)

ESP32 (Optional hardware integration)

📊 Features

✔ Real-time crowd detection
✔ Person counting using YOLO
✔ Intelligent risk assessment
✔ Hybrid soft computing model
✔ Dashboard visualization
✔ Sensor integration (optional)

🎯 Applications

Public events monitoring

Stadium crowd control

Religious gatherings

Railway stations

Emergency evacuation systems