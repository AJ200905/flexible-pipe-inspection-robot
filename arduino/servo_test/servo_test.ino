// servo_test.ino - Goteck GD2508MG on GPIO 23
// Starts at neutral (1520 us). In miniterm, type a pulse width 500..2500 and press Enter.
const int SERVO_PIN  = 23;
const int FREQ_HZ    = 333;                      // GD2508MG: best at 333 Hz (50 Hz also works)
const int NEUTRAL_US = 1520;
const float PERIOD_US = 1000000.0 / FREQ_HZ;     // 3003 us

void writeUs(int us) {
  us = constrain(us, 500, 2500);
  ledcWrite(SERVO_PIN, (uint32_t)(us / PERIOD_US * 16383.0));
  Serial.printf("Servo pulse = %d us\n", us);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  if (!ledcAttach(SERVO_PIN, FREQ_HZ, 14)) Serial.println("PWM setup FAILED");
  writeUs(NEUTRAL_US);
  Serial.println("At neutral. Type a pulse in us (500-2500), e.g. 1520, 1300, 1700");
}

void loop() {
  if (Serial.available()) {
    int us = Serial.parseInt();
    while (Serial.available()) Serial.read();    // clear the rest of the line
    if (us >= 500 && us <= 2500) writeUs(us);
  }
}
