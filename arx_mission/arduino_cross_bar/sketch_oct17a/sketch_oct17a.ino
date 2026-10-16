// Cross bar: MG996R servo + HC-SR04 ultrasonic
// เริ่มต้น servo อยู่ที่ 0 องศา
// เมื่อ ultrasonic เจอหุ่นผ่าน (ระยะ < 20 cm) -> เปิดไม้กั้น 90 องศา ค้างไว้ 5 วินาที แล้วปิดกลับ 0 องศา

#include <Servo.h>

// ---- Pins ----
const int SERVO_PIN = 9;   // สายสัญญาณ (ส้ม) ของ MG996R
const int TRIG_PIN  = 7;   // HC-SR04 Trig
const int ECHO_PIN  = 6;   // HC-SR04 Echo

// ---- Settings ----
const int   REST_ANGLE      = 0;      // ปิดไม้กั้น (มุมเริ่มต้น)
const int   ACTIVE_ANGLE    = 90;     // เปิดไม้กั้นตอนเจอหุ่น
const float DETECT_CM       = 20.0;   // ระยะตรวจจับ
const unsigned long HOLD_MS = 5000;   // ค้างไว้ 5 วินาที
const int   CONFIRM_COUNT   = 3;     // ต้องเจอติดกันกี่ครั้งถึงนับว่าเจอ (กันค่ากระโดด)

Servo bar;
int hitCount = 0;

// อ่านระยะเป็น cm, คืนค่า -1 ถ้าไม่มี echo (เกินระยะ)
float readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);  // timeout ~5 m
  if (duration == 0) return -1;
  return duration * 0.0343 / 2.0;
}

void setup() {
  Serial.begin(9600);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  bar.attach(SERVO_PIN);
  bar.write(REST_ANGLE);
  delay(500);
  Serial.println("Cross bar ready (servo at 0 deg)");
}

void loop() {
  float d = readDistanceCm();
  Serial.print("Distance: ");
  Serial.print(d);
  Serial.println(" cm");

  if (d > 0 && d < DETECT_CM) {
    hitCount++;
  } else {
    hitCount = 0;
  }

  if (hitCount >= CONFIRM_COUNT) {
    Serial.println("Robot detected -> hold 5 s");
    bar.write(ACTIVE_ANGLE);
    delay(HOLD_MS);
    bar.write(REST_ANGLE);
    Serial.println("Back to 0 deg");
    delay(500);   // ให้ servo กลับถึงที่ก่อนวัดใหม่
    hitCount = 0;
  }

  delay(60);  // HC-SR04 ควรเว้น >= 60 ms ระหว่างการวัด
}
