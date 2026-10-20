// center_servo.ino - holds the servo on GPIO 23 at its midpoint (1500 us = 90 deg)
const int SERVO_PIN = 23;

void setup() {
  Serial.begin(115200);
  ledcAttach(SERVO_PIN, 50, 14);                      // 50 Hz, 14-bit (ESP32 core 3.x)
  uint32_t duty = 1500UL * 16383UL / 20000UL;         // 1500 us out of a 20 ms period
  ledcWrite(SERVO_PIN, duty);
  Serial.println("Servo at midpoint (1500 us)");
}

void loop() {}                                        // keeps holding the position
