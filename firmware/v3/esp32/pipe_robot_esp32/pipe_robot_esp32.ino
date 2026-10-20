/*
  ============================================================
   PIPE INSPECTION ROBOT  -  ESP32 FIRMWARE   (v3: TB6612 + GD2508MG tendon)
  ============================================================
   Controls : 3 DC drive motors on 2x TB6612FNG drivers,
              addressable LED (WS2812-type),
              Goteck GD2508MG servo that pulls the tendon to bend the TPU linkage,
              MPU6050 IMU
   Talks to : Raspberry Pi over USB serial (115200 baud)
   Board    : "ESP32 Dev Module"  (ESP32 Arduino core v3.x)
   Libraries: Adafruit NeoPixel (install once), Wire (built in)

   COMMANDS (one per line, sent by the Pi)
   ---------------------------------------
   P                  ping                          -> replies "OK"
   F 60 / B 60        all motors forward / backward at 60 %
   S                  stop (brake)
   M 50 50 50         each motor separately (-100..100 %)
   L 80               LED brightness 0..100 %
   D 40               bend the linkage: -100 (full right) .. 0 (straight) .. +100 (full left)
   V 1450             raw servo pulse in microseconds (CALIBRATION ONLY, clamped to safe limits)
   Z                  zero the IMU (take current pose as level)
   K S 1.0 6.0 0.0    set SPEED PID gains (kp ki kd)  - only used with encoders
   G                  print current settings

   TELEMETRY (sent 20 times/second)
   --------------------------------
   T,roll,pitch,cmd1,cmd2,cmd3,rpm1,rpm2,rpm3,light,servo_us

   SAFETY : motors stop if no command arrives for 1 second.
            The bend is kept (the robot does not snap straight inside an elbow).
   SERVO  : always starts at mid position (1520 us), then moves with smooth
            acceleration / deceleration (trapezoidal profile) - no jerks.
  ============================================================
*/

#include <Wire.h>
#include <math.h>
#include <Adafruit_NeoPixel.h>

// ================== 1. SETTINGS YOU MAY CHANGE ==================

// ---- Motor driver pins: 2x TB6612FNG ----
//              Motor 1 (A: AIN1/AIN2/PWMA), Motor 2 (A: BIN1/BIN2/PWMB), Motor 3 (B: AIN1/AIN2/PWMA)
const int M_IN1[3] = {26, 14, 17};
const int M_IN2[3] = {25, 13, 16};
const int M_EN [3] = {27, 33, 18};      // PWM speed pins
const bool M_REVERSE[3] = {false, false, false};  // set true if that motor spins the wrong way
// NOTE: STBY pin on BOTH TB6612 boards must be tied to 3.3 V (HIGH), or nothing moves.

const int PWM_FREQ = 20000;  // 20 kHz: right for TB6612, and silent
const int PWM_BITS = 10;     // 0..1023
const int PWM_MAX  = (1 << PWM_BITS) - 1;
const int MIN_PWM_PCT = 25;  // dead zone: below this the motors only hum. Tune it.

// ---- LED (WS2812-type, DI pin) ----
const int LIGHT_PIN     = 19;   // -> 330 ohm -> LED DI
const int LED_COUNT     = 1;    // number of LEDs (change for a ring/strip)
const int LIGHT_ON_BOOT = 100;  // % brightness at power-up

// ---- Tendon servo: Goteck GD2508MG ----
const int   SERVO_PIN         = 23;
const int   SERVO_FREQ_HZ     = 333;    // GD2508MG: 333 Hz = best accuracy and holding torque
const int   SERVO_MID_US      = 1520;   // servo neutral - ALWAYS used at power-up
const int   SERVO_STRAIGHT_US = 1520;   // pulse where the TPU linkage is straight (calibrate)
const int   SERVO_BEND_US     = 200;    // pulse change from straight for FULL bend (calibrate)
const int   SERVO_SAFE_MIN_US = 1300;   // hard limits - never go beyond (protect tendon + TPU)
const int   SERVO_SAFE_MAX_US = 1740;
const bool  SERVO_INVERT      = false;  // true if D + (left) bends the robot to the right
const float SERVO_MAX_SPEED   = 800.0;  // us per second   (top speed of the profile)
const float SERVO_ACCEL       = 2000.0; // us per second^2 (how gently it speeds up / slows down)

// ---- I2C (MPU6050) ----
const int SDA_PIN = 21;
const int SCL_PIN = 22;

// ---- Encoders (set USE_ENCODERS to 1 only if your motors have them) ----
#define USE_ENCODERS 0
const int   ENC_PIN[3] = {34, 35, 39};
const float ENC_CPR    = 700.0;
const float MAX_RPM    = 100.0;

// ---- Speed PID (used only with encoders) ----
float SPD_KP = 1.0, SPD_KI = 6.0, SPD_KD = 0.0;
const float SLOPE_FF = 40.0;   // extra % per unit sin(pitch) when climbing (0 = off)

// ---- Timing ----
const unsigned long CONTROL_US   = 10000;  // control loop 100 Hz
const unsigned long TELEMETRY_MS = 50;     // telemetry 20 Hz
const unsigned long WATCHDOG_MS  = 1000;   // stop motors if Pi goes silent

// ================== 2. SMALL PID HELPER ==================
struct PID {
  float kp, ki, kd;
  float integ = 0, prevMeas = 0;
  float outMin, outMax;
  bool  first = true;
  void reset() { integ = 0; first = true; }
  float update(float target, float meas, float dt) {
    float err = target - meas;
    float dMeas = first ? 0 : (meas - prevMeas) / dt;
    first = false;
    prevMeas = meas;
    float newInteg = integ + ki * err * dt;
    float out = kp * err + newInteg - kd * dMeas;
    if (out > outMax)      out = outMax;
    else if (out < outMin) out = outMin;
    else                   integ = newInteg;     // anti-windup
    return out;
  }
};
PID speedPid[3];

// ================== 3. STATE ==================
float target[3]   = {0, 0, 0};
float motorCmd[3] = {0, 0, 0};
float rpm[3]      = {0, 0, 0};
int   lightPct = 0;

float servoUs       = SERVO_MID_US;   // where the servo is now
float servoVel      = 0;              // current speed (us/s), for smooth motion
float servoTargetUs = SERVO_MID_US;   // where it is going
int   bendPct       = 0;

float roll = 0, pitch = 0;
float rollZero = 0, pitchZero = 0;
float gyroBiasX = 0, gyroBiasY = 0;
float rawRoll = 0, rawPitch = 0;
bool  imuOk = false;

unsigned long lastCmdMs = 0, lastCtrlUs = 0, lastTelMs = 0;
String rxLine = "";

#if USE_ENCODERS
volatile long encCount[3] = {0, 0, 0};
void IRAM_ATTR enc0() { encCount[0]++; }
void IRAM_ATTR enc1() { encCount[1]++; }
void IRAM_ATTR enc2() { encCount[2]++; }
long lastCount[3] = {0, 0, 0};
#endif

Adafruit_NeoPixel leds(LED_COUNT, LIGHT_PIN, NEO_GRB + NEO_KHZ800);

// ================== 4. MOTORS ==================
void setMotor(int i, float pct) {
  pct = constrain(pct, -100, 100);
  motorCmd[i] = pct;
  float p = M_REVERSE[i] ? -pct : pct;
  if (fabs(p) < 1.0) {                       // stop = short brake (TB6612: IN1=IN2=HIGH)
    digitalWrite(M_IN1[i], HIGH);
    digitalWrite(M_IN2[i], HIGH);
    ledcWrite(M_EN[i], 0);
    return;
  }
  float mag = MIN_PWM_PCT + (100.0 - MIN_PWM_PCT) * fabs(p) / 100.0;
  digitalWrite(M_IN1[i], p > 0 ? HIGH : LOW);
  digitalWrite(M_IN2[i], p > 0 ? LOW : HIGH);
  ledcWrite(M_EN[i], (int)(mag / 100.0 * PWM_MAX));
}

void stopAll() {
  for (int i = 0; i < 3; i++) {
    target[i] = 0;
    speedPid[i].reset();
    setMotor(i, 0);
  }
}

// ================== 5. LED ==================
void setLight(int pct) {
  lightPct = constrain(pct, 0, 100);
  uint8_t v = lightPct * 255 / 100;
  leds.fill(leds.Color(v, v, v));
  leds.show();
}

// ================== 6. SERVO (smooth motion) ==================
void writeServoUs(float us) {
  const float periodUs = 1000000.0 / SERVO_FREQ_HZ;      // 3003 us at 333 Hz
  ledcWrite(SERVO_PIN, (uint32_t)(us / periodUs * 16383.0));
}

void setServoTarget(float us) {
  servoTargetUs = constrain(us, SERVO_SAFE_MIN_US, SERVO_SAFE_MAX_US);
}

// bend -100 (right) .. +100 (left)
void setBend(int pct) {
  bendPct = constrain(pct, -100, 100);
  float dir = SERVO_INVERT ? -1.0 : 1.0;
  setServoTarget(SERVO_STRAIGHT_US + dir * bendPct * SERVO_BEND_US / 100.0);
}

// Trapezoidal profile: accelerate, cruise at max speed, decelerate to stop exactly on target
void servoStep(float dt) {
  float err = servoTargetUs - servoUs;
  if (fabs(err) < 0.5 && fabs(servoVel) < 20.0) {         // arrived
    servoVel = 0;
    if (servoUs != servoTargetUs) { servoUs = servoTargetUs; writeServoUs(servoUs); }
    return;
  }
  float dir = err > 0 ? 1.0 : -1.0;
  // fastest speed from which we can still stop on target: v = sqrt(2 * a * distance)
  float wantVel = dir * min(SERVO_MAX_SPEED, sqrtf(2.0 * SERVO_ACCEL * fabs(err)));

  float dvMax = SERVO_ACCEL * dt;                         // limit acceleration
  servoVel += constrain(wantVel - servoVel, -dvMax, dvMax);

  float step = servoVel * dt;
  if ((dir > 0 && step > err) || (dir < 0 && step < err)) { // never overshoot
    step = err;
    servoVel = 0;
  }
  servoUs += step;
  writeServoUs(servoUs);
}

// ================== 7. IMU (MPU6050, no library) ==================
const uint8_t MPU = 0x68;

bool imuBegin() {
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.beginTransmission(MPU);
  Wire.write(0x6B); Wire.write(0x00);          // wake up
  if (Wire.endTransmission() != 0) return false;
  Wire.beginTransmission(MPU);
  Wire.write(0x1A); Wire.write(0x03);          // 44 Hz low-pass filter
  Wire.endTransmission();
  return true;
}

bool imuRead(float &ax, float &ay, float &az, float &gx, float &gy) {
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU, (uint8_t)14) != 14) return false;
  int16_t raw[7];
  for (int i = 0; i < 7; i++) raw[i] = (Wire.read() << 8) | Wire.read();
  ax = raw[0] / 16384.0;  ay = raw[1] / 16384.0;  az = raw[2] / 16384.0;
  gx = raw[4] / 131.0;    gy = raw[5] / 131.0;
  return true;
}

void imuCalibrateGyro() {               // keep the robot still for ~1 s at power-up
  float sx = 0, sy = 0, ax, ay, az, gx, gy;
  int n = 0;
  for (int i = 0; i < 200; i++) {
    if (imuRead(ax, ay, az, gx, gy)) { sx += gx; sy += gy; n++; }
    delay(5);
  }
  if (n) { gyroBiasX = sx / n; gyroBiasY = sy / n; }
}

void imuUpdate(float dt) {              // complementary filter
  float ax, ay, az, gx, gy;
  if (!imuRead(ax, ay, az, gx, gy)) return;
  float accRoll  = atan2(ay, az) * 180.0 / PI;
  float accPitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
  const float A = 0.98;
  rawRoll  = A * (rawRoll  + (gx - gyroBiasX) * dt) + (1 - A) * accRoll;
  rawPitch = A * (rawPitch + (gy - gyroBiasY) * dt) + (1 - A) * accPitch;
  roll  = rawRoll  - rollZero;
  pitch = rawPitch - pitchZero;
}

// ================== 8. ENCODERS ==================
void encoderUpdate(float dt) {
#if USE_ENCODERS
  for (int i = 0; i < 3; i++) {
    noInterrupts();
    long c = encCount[i];
    interrupts();
    float r = (c - lastCount[i]) / ENC_CPR / dt * 60.0;
    lastCount[i] = c;
    if (motorCmd[i] < 0) r = -r;
    rpm[i] = 0.7 * rpm[i] + 0.3 * r;
  }
#endif
}

// ================== 9. CONTROL LOOP (100 Hz) ==================
void controlStep(float dt) {
  if (imuOk) imuUpdate(dt);
  encoderUpdate(dt);
  servoStep(dt);                                       // smooth tendon motion

  bool moving = fabs(target[0]) + fabs(target[1]) + fabs(target[2]) > 0.5;
  float slope = moving ? SLOPE_FF * sin(pitch * PI / 180.0) : 0;

  for (int i = 0; i < 3; i++) {
    if (!moving) { setMotor(i, 0); continue; }
#if USE_ENCODERS
    float corr = speedPid[i].update(target[i], rpm[i] / MAX_RPM * 100.0, dt);
    setMotor(i, target[i] + slope + corr);
#else
    setMotor(i, target[i] + slope);                    // open loop
#endif
  }
}

// ================== 10. COMMANDS ==================
void applyGains() {
  for (int i = 0; i < 3; i++) {
    speedPid[i].kp = SPD_KP; speedPid[i].ki = SPD_KI; speedPid[i].kd = SPD_KD;
    speedPid[i].outMin = -50; speedPid[i].outMax = 50;
    speedPid[i].reset();
  }
}

void printSettings() {
  Serial.printf("# speed PID kp=%.2f ki=%.2f kd=%.2f | encoders=%d imu=%d light=%d\n",
                SPD_KP, SPD_KI, SPD_KD, USE_ENCODERS, imuOk, lightPct);
  Serial.printf("# servo now=%.0f us target=%.0f us bend=%d%% | straight=%d bend=+-%d safe=%d..%d invert=%d\n",
                servoUs, servoTargetUs, bendPct, SERVO_STRAIGHT_US, SERVO_BEND_US,
                SERVO_SAFE_MIN_US, SERVO_SAFE_MAX_US, SERVO_INVERT);
}

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) return;
  lastCmdMs = millis();
  char c = toupper(line.charAt(0));
  String args = line.substring(1);
  args.trim();
  float a = 0, b = 0, d = 0;

  switch (c) {
    case 'P': Serial.println("OK"); break;
    case 'F': a = args.toFloat(); for (int i = 0; i < 3; i++) target[i] =  constrain(a, 0, 100); break;
    case 'B': a = args.toFloat(); for (int i = 0; i < 3; i++) target[i] = -constrain(a, 0, 100); break;
    case 'S': stopAll(); break;
    case 'M':
      if (sscanf(args.c_str(), "%f %f %f", &a, &b, &d) == 3) {
        target[0] = constrain(a, -100, 100);
        target[1] = constrain(b, -100, 100);
        target[2] = constrain(d, -100, 100);
      }
      break;
    case 'L': setLight(args.toInt()); break;
    case 'D': setBend(args.toInt()); break;
    case 'V': bendPct = 0; setServoTarget(args.toInt()); break;   // raw us, calibration
    case 'Z': rollZero = rawRoll; pitchZero = rawPitch; Serial.println("# IMU zeroed"); break;
    case 'G': printSettings(); break;
    case 'K': {
      char which; float kp, ki, kd;
      if (sscanf(args.c_str(), "%c %f %f %f", &which, &kp, &ki, &kd) == 4 && toupper(which) == 'S') {
        SPD_KP = kp; SPD_KI = ki; SPD_KD = kd;
        applyGains();
        printSettings();
      }
      break;
    }
    default: Serial.println("# unknown command");
  }
}

void readSerial() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (rxLine.length()) handleCommand(rxLine);
      rxLine = "";
    } else if (rxLine.length() < 60) {
      rxLine += ch;
    }
  }
}

void sendTelemetry() {
  Serial.printf("T,%.1f,%.1f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%d,%d\n",
                roll, pitch, motorCmd[0], motorCmd[1], motorCmd[2],
                rpm[0], rpm[1], rpm[2], lightPct, (int)servoUs);
}

// ================== 11. SETUP & LOOP ==================
void setup() {
  // servo FIRST: hold mid position from the very start
  ledcAttach(SERVO_PIN, SERVO_FREQ_HZ, 14);
  writeServoUs(SERVO_MID_US);

  Serial.begin(115200);

  for (int i = 0; i < 3; i++) {
    pinMode(M_IN1[i], OUTPUT);
    pinMode(M_IN2[i], OUTPUT);
    ledcAttach(M_EN[i], PWM_FREQ, PWM_BITS);
  }
  leds.begin();

#if USE_ENCODERS
  for (int i = 0; i < 3; i++) pinMode(ENC_PIN[i], INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_PIN[0]), enc0, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_PIN[1]), enc1, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_PIN[2]), enc2, RISING);
#endif

  applyGains();
  stopAll();
  setLight(LIGHT_ON_BOOT);

  imuOk = imuBegin();
  if (imuOk) imuCalibrateGyro();

  setBend(0);   // then glide smoothly from mid to 'straight' (same if both are 1520)

  Serial.println(imuOk ? "# READY (IMU ok)" : "# READY (IMU NOT FOUND - check wiring)");
  lastCmdMs = millis();
  lastCtrlUs = micros();
}

void loop() {
  readSerial();

  if (millis() - lastCmdMs > WATCHDOG_MS) {
    bool anyMoving = target[0] != 0 || target[1] != 0 || target[2] != 0;
    if (anyMoving) { stopAll(); Serial.println("# watchdog stop"); }
  }

  unsigned long now = micros();
  if (now - lastCtrlUs >= CONTROL_US) {
    float dt = (now - lastCtrlUs) / 1e6;
    lastCtrlUs = now;
    controlStep(dt);
  }

  if (millis() - lastTelMs >= TELEMETRY_MS) {
    lastTelMs = millis();
    sendTelemetry();
  }
}
