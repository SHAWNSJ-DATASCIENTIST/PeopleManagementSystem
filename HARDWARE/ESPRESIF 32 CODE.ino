#include <Wire.h>
#include <ESP32Servo.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <SimpleKalmanFilter.h>
#include <ArduinoJson.h>

Servo myServo;
Adafruit_MPU6050 mpu;

SimpleKalmanFilter kalmanX(0.5, 0.5, 0.001);
SimpleKalmanFilter kalmanY(0.5, 0.5, 0.001);
SimpleKalmanFilter kalmanZ(0.5, 0.5, 0.001);

int servoPin = 18;

int ir1 = 34;
int ir2 = 35;

int trigPin = 5;
int echoPin = 19;

int p1Pin = 32;
int p2Pin = 33

int ledPin = 2;

float tremorThreshold = 0.08;
int piezoThreshold = 1500;

long duration;
float distance;

float baseX = 0, baseY = 0, baseZ = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  myServo.attach(servoPin);

  pinMode(ir1, INPUT);
  pinMode(ir2, INPUT);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(p1Pin, INPUT);
  pinMode(p2Pin, INPUT);
  pinMode(ledPin, OUTPUT);

  mpu.begin();
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);

  calibrate();
}

void loop() {

  for (int pos = 0; pos <= 180; pos++) {
    myServo.write(pos);
    processAll();
    delay(3);
  }

  for (int pos = 180; pos >= 0; pos--) {
    myServo.write(pos);
    processAll();
    delay(3);
  }
}

void calibrate() {
  sensors_event_t a, g, temp;
  for (int i = 0; i < 200; i++) {
    mpu.getEvent(&a, &g, &temp);
    baseX += a.acceleration.x;
    baseY += a.acceleration.y;
    baseZ += a.acceleration.z;
    delay(5);
  }
  baseX /= 200;
  baseY /= 200;
  baseZ /= 200;
}

void blinkTwice() {
  for (int i = 0; i < 2; i++) {
    digitalWrite(ledPin, HIGH);
    delay(120);
    digitalWrite(ledPin, LOW);
    delay(120);
  }
}

void processAll() {

  int ir1Val = (digitalRead(ir1) == LOW) ? 1 : 0;
  int ir2Val = (digitalRead(ir2) == LOW) ? 1 : 0;

  int p1Raw = analogRead(p1Pin);
  int p2Raw = analogRead(p2Pin);

  int p1Val = (p1Raw > piezoThreshold) ? 1 : 0;
  int p2Val = (p2Raw > piezoThreshold) ? 1 : 0;

  if (p1Val || p2Val) {
    blinkTwice();
  }

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  float fx = kalmanX.updateEstimate(a.acceleration.x - baseX);
  float fy = kalmanY.updateEstimate(a.acceleration.y - baseY);
  float fz = kalmanZ.updateEstimate(a.acceleration.z - baseZ);

  float vibration = sqrt(fx * fx + fy * fy + fz * fz);
  int vib = (vibration > tremorThreshold) ? 1 : 0;

  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH, 20000);
  distance = duration * 0.034 / 2;

  StaticJsonDocument<256> doc;

  doc["D"]   = serialized(String(distance, 1));
  doc["V"]   = vib;
  doc["IR"]  = ir1Val;
  doc["IR2"] = ir2Val;

  doc["PIR"] = 0;
  doc["P1"]  = p1Val;
  doc["P2"]  = p2Val;
  doc["P3"]  = 0;
  doc["P4"]  = 0;
  doc["P5"]  = 0;

  serializeJson(doc, Serial);
  Serial.println();
}