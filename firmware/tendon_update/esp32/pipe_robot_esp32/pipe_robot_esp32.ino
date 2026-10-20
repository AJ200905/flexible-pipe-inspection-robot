/*
  ============================================================
   PIPE INSPECTION ROBOT  -  ESP32 FIRMWARE
  ============================================================
   Controls : 3 DC drive motors (2 drivers), addressable LED, tendon-steering
              servo (bends the TPU linkage left/right), MPU6050 IMU
   Talks to : Raspberry Pi over USB serial (115200 baud)
   Board    : "ESP32 Dev Module"  (ESP32 board package v3.x)
   Libraries: Adafruit NeoPixel (install once), Wire (built in)

   COMMANDS (one per line, sent by the Pi)
   ---------------------------------------
   P                  ping                   -> replies "OK"
   F 60               forward at 60 %
   B 60               backward at 60 %
   S                  stop (brake)
   M 50 50 -20        set each motor separately (-100..100 %)
   L 80               light brightness 0..100 %
   D 40               bend the tendon linkage: -100 (full right) .. 0 (straight)
                      .. +100 (full left). Servo moves smoothly (slew-limited).
   V 90               raw servo angle 0..180 deg (only for calibration!)
   Z                  zero the IMU (take current pose as level)
   K S 1.0 6.0 0.0    set SPEED PID gains  (kp ki kd)
   G                  print current settings

   TELEMETRY (sent 20 times/second)
   --------------------------------
   T,roll,pitch,cmd1,cmd2,cmd3,rpm1,rpm2,rpm3,light,servo_deg
   (angles in degrees, cmd = motor output -100..100 %)

   SAFETY: motors stop if no command arrives for 1 second.
           The bend is kept (so the robot does not snap straight inside an elbow).
  ============================================================
*/

#include <Wire.h>
#include <math.h>
#include <Adafruit_NeoPixel.h>   // Library: "Adafruit NeoPixel"

// ================== 1. SETTINGS YOU MAY CHANGE ==================

// ---- Motor driver pins (L298N or TB6612: IN1, IN2, EN/PWM) ----
// Driver A -> Motor 1 and Motor 2 ;  Driver B -> Motor 3
const int M_IN1[3] = {25, 14, 16};
const int M_IN2[3] = {26, 13, 17};
const int M_EN [3] = {27, 33, 18};      // PWM speed pins
const bool M_REVERSE[3] = {false, false, false};  // flip if a motor spins the wrong way

const int LIGHT_PIN = 19;   // -> 330 ohm -> LED DI (addressable WS2812-type LED)
const int LED_COUNT = 1;    // number of LEDs on the strip / ring (change to match yours)
const int LIGHT_ON_BOOT = 100;  // % brightness at power-up (100 = full white)
const int SERVO_PIN = 23;

// ---- Tendon steering servo (calibrate these!) ----
const int   SERVO_CENTER_DEG  = 90;    // servo angle where the TPU linkage is STRAIGHT
const int   SERVO_MAX_BEND_DEG = 60;   // servo travel from centre for FULL bend (protects tendon + TPU)
const bool  SERVO_INVERT      = false; // true if "left" (D +) bends the robot to the right
const float SERVO_SLEW_DEG_S  = 90.0;  // max servo speed in deg/s - gentle on the tendon
const int SDA_PIN   = 21;
const int SCL_PIN   = 22;

// ---- PWM ----
const int PWM_FREQ = 1000;  // 1 kHz suits L298N. Use 20000 for TB6612.
const int PWM_BITS = 10;    // 0..1023
const int PWM_MAX  = (1 << PWM_BITS) - 1;
const int MIN_PWM_PCT = 25; // below this the motors usually only hum (dead zone). Tune it.

// ---- Encoders (set USE_ENCODERS to 1 only if your motors have them) ----
#define USE_ENCODERS 0
const int   ENC_PIN[3] = {34, 35, 39};  // input-only pins: encoder board must have pull-ups
const float ENC_CPR    = 700.0;         // pulses per OUTPUT shaft turn (check motor datasheet)
const float MAX_RPM    = 100.0;         // motor rated speed after gearbox; 100 % = this

// ---- PID starting gains (change live with the K command) ----
float SPD_KP = 1.0, SPD_KI = 6.0, SPD_KD = 0.0;   // speed loop  (error in %, output in %)
const float SLOPE_FF = 40.0;   // extra % per unit sin(pitch) to climb slopes (0 = off)

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

  // derivative on measurement -> no "kick" when the target jumps
  float update(float target, float meas, float dt) {
    float err = target - meas;
    float dMeas = first ? 0 : (meas - prevMeas) / dt;
    first = false;
    prevMeas = meas;
    float pTerm = kp * err;
    float dTerm = -kd * dMeas;
    float newInteg = integ + ki * err * dt;
    float out = pTerm + newInteg + dTerm;
    // anti-windup: only keep integrating when not saturated
    if (out > outMax)      { out = outMax; }
    else if (out < outMin) { out = outMin; }
    else                   { integ = newInteg; }
    return out;
  }
};

PID speedPid[3];

// ================== 3. STATE ==================
float target[3]   = {0, 0, 0};   // what the Pi asked for (-100..100 %)
float motorCmd[3] = {0, 0, 0};   // what we actually send (-100..100 %)
float rpm[3]      = {0, 0, 0};
int   lightPct = 0;
float servoNow    = SERVO_CENTER_DEG;   // where the servo is now (deg)
float servoTarget = SERVO_CENTER_DEG;   // where it is going (deg)
int   bendPct = 0;                      // -100 right .. +100 left

float roll = 0, pitch = 0;               // degrees, after zeroing
float rollZero = 0, pitchZero = 0;
float gyroBiasX = 0, gyroBiasY = 0;
bool  imuOk = false;

unsigned long lastCmdMs = 0;
unsigned long lastCtrlUs = 0;
unsigned long lastTelMs = 0;
String rxLine = "";

#if USE_ENCODERS
volatile long encCount[3] = {0, 0, 0};
void IRAM_ATTR enc0() { encCount[0]++; }
void IRAM_ATTR enc1() { encCount[1]++; }
void IRAM_ATTR enc2() { encCount[2]++; }
long lastCount[3] = {0, 0, 0};
#endif

// ================== 4. MOTORS, LIGHT, SERVO ==================
void setMotor(int i, float pct) {
  pct = constrain(pct, -100, 100);
  motorCmd[i] = pct;
  float p = M_REVERSE[i] ? -pct : pct;

  if (fabs(p) < 1.0) {                 // stop = brake
    digitalWrite(M_IN1[i], HIGH);
    digitalWrite(M_IN2[i], HIGH);
    ledcWrite(M_EN[i], 0);
    return;
  }
  // map 1..100 % onto MIN_PWM_PCT..100 % so small commands still move
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

Adafruit_NeoPixel leds(LED_COUNT, LIGHT_PIN, NEO_GRB + NEO_KHZ800);

void setLight(int pct) {
  lightPct = constrain(pct, 0, 100);
  uint8_t v = lightPct * 255 / 100;          // white = R, G, B all equal
  leds.fill(leds.Color(v, v, v));
  leds.show();
}

// servo: 50 Hz, 14-bit -> 16383 counts per 20 ms
void writeServo(float deg) {
  deg = constrain(deg, 0, 180);
  int us = 500 + (int)(deg / 180.0 * 2000.0);
  ledcWrite(SERVO_PIN, (uint32_t)us * 16383UL / 20000UL);
}

// tendon bend: -100 (full right) .. +100 (full left)
void setBend(int pct) {
  bendPct = constrain(pct, -100, 100);
  float dir = SERVO_INVERT ? -1.0 : 1.0;
  servoTarget = SERVO_CENTER_DEG + dir * bendPct * SERVO_MAX_BEND_DEG / 100.0;
}

// move the servo towards its target, no faster than SERVO_SLEW_DEG_S
void servoStep(float dt) {
  float step = SERVO_SLEW_DEG_S * dt;
  float diff = servoTarget - servoNow;
  if (fabs(diff) < 0.05) return;
  servoNow += constrain(diff, -step, step);
  writeServo(servoNow);
}

// ================== 5. IMU (MPU6050, no library) ==================
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
  ax = raw[0] / 16384.0;  ay = raw[1] / 16384.0;  az = raw[2] / 16384.0;  // g
  gx = raw[4] / 131.0;    gy = raw[5] / 131.0;                             // deg/s
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

// complementary filter: gyro for fast changes, accelerometer for long-term truth
float rawRoll = 0, rawPitch = 0;
void imuUpdate(float dt) {
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

// ================== 6. ENCODERS ==================
void encoderUpdate(float dt) {
#if USE_ENCODERS
  for (int i = 0; i < 3; i++) {
    noInterrupts();
    long c = encCount[i];
    interrupts();
    float turns = (c - lastCount[i]) / ENC_CPR;
    lastCount[i] = c;
    float r = turns / dt * 60.0;
    if (motorCmd[i] < 0) r = -r;             // single channel -> sign from command
    rpm[i] = 0.7 * rpm[i] + 0.3 * r;         // light smoothing
  }
#endif
}

// ================== 7. CONTROL LOOP (100 Hz) ==================
void controlStep(float dt) {
  if (imuOk) imuUpdate(dt);
  encoderUpdate(dt);

  bool moving = fabs(target[0]) + fabs(target[1]) + fabs(target[2]) > 0.5;

  // (a) tendon steering servo - smooth move towards the requested bend
  servoStep(dt);

  // (b) slope feed-forward: gravity pulls back by ~sin(pitch)
  float slope = moving ? SLOPE_FF * sin(pitch * PI / 180.0) : 0;

  // (c) speed loop for each motor (all 3 push along the pipe axis)
  for (int i = 0; i < 3; i++) {
    if (!moving) { setMotor(i, 0); continue; }
    float want = target[i];
#if USE_ENCODERS
    float measPct = rpm[i] / MAX_RPM * 100.0;
    // feed-forward (want) + PID correction
    float corr = speedPid[i].update(want, measPct, dt);
    setMotor(i, want + slope + corr);
#else
    setMotor(i, want + slope);         // no encoders -> open loop
#endif
  }
}

// ================== 8. COMMAND PARSER ==================
void printSettings() {
  Serial.printf("# speed PID kp=%.3f ki=%.3f kd=%.3f\n", SPD_KP, SPD_KI, SPD_KD);
  Serial.printf("# bend=%d%% servo=%.0f deg (centre %d, max bend +-%d, invert %d)\n",
                bendPct, servoNow, SERVO_CENTER_DEG, SERVO_MAX_BEND_DEG, SERVO_INVERT);
  Serial.printf("# encoders=%d imu=%d light=%d\n", USE_ENCODERS, imuOk, lightPct);
}

void applyGains() {
  for (int i = 0; i < 3; i++) {
    speedPid[i].kp = SPD_KP; speedPid[i].ki = SPD_KI; speedPid[i].kd = SPD_KD;
    speedPid[i].outMin = -50; speedPid[i].outMax = 50;
    speedPid[i].reset();
  }
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
    case 'V':                                      // raw angle, for calibration only
      servoTarget = constrain(args.toInt(), 0, 180);
      bendPct = 0;
      break;
    case 'R': Serial.println("# roll modes removed (servo is now tendon steering)"); break;
    case 'Z': rollZero = rawRoll; pitchZero = rawPitch; Serial.println("# IMU zeroed"); break;
    case 'G': printSettings(); break;
    case 'K': {
      char which; float kp, ki, kd;
      if (sscanf(args.c_str(), "%c %f %f %f", &which, &kp, &ki, &kd) == 4) {
        which = toupper(which);
        if (which == 'S') { SPD_KP = kp; SPD_KI = ki; SPD_KD = kd; }
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
                rpm[0], rpm[1], rpm[2], lightPct, (int)servoNow);
}

// ================== 9. SETUP & LOOP ==================
void setup() {
  Serial.begin(115200);

  for (int i = 0; i < 3; i++) {
    pinMode(M_IN1[i], OUTPUT);
    pinMode(M_IN2[i], OUTPUT);
    ledcAttach(M_EN[i], PWM_FREQ, PWM_BITS);
  }
  leds.begin();
  ledcAttach(SERVO_PIN, 50, 14);

#if USE_ENCODERS
  for (int i = 0; i < 3; i++) pinMode(ENC_PIN[i], INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_PIN[0]), enc0, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_PIN[1]), enc1, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_PIN[2]), enc2, RISING);
#endif

  applyGains();
  stopAll();
  setLight(LIGHT_ON_BOOT);
  writeServo(servoNow);                  // start straight (SERVO_CENTER_DEG)

  imuOk = imuBegin();
  if (imuOk) imuCalibrateGyro();

  Serial.println(imuOk ? "# READY (IMU ok)" : "# READY (IMU NOT FOUND - check wiring)");
  lastCmdMs = millis();
  lastCtrlUs = micros();
}

void loop() {
  readSerial();

  // safety watchdog
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
