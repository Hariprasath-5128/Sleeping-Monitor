// ==========================================
// Sleeping Monitor — Motor Control via Local Server
// ESP32 polls Flask /status every 2 s and drives
// servos based on the zone reported by img_process.py
// ==========================================

#include <ESP32Servo.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ──────────────────────────────────────────
// WiFi credentials  ← must match the ESP32-CAM and your PC
// ──────────────────────────────────────────
const char* WIFI_SSID     = "Hariprasath LAPTOP";
const char* WIFI_PASSWORD = "123456789";

// ──────────────────────────────────────────
// Server address  ← set to your PC's LAN IP
// ──────────────────────────────────────────
const char* SERVER_IP = "192.168.137.1";
const int   SERVER_PORT = 5000;

// Poll interval (ms)
const unsigned long POLL_INTERVAL_MS = 250;

// ──────────────────────────────────────────
// Servo config — UNCHANGED from original
// ──────────────────────────────────────────
// ONE MOTOR PER SIDE.
//   GP19 -> LEFT  edge of the bed
//   GP18 -> RIGHT edge of the bed
// Named by side, not by number, so the mapping cannot be misread.
Servo servoLeft;    // GP19
Servo servoRight;   // GP18

const int PIN_LEFT   = 19;
const int PIN_RIGHT  = 18;

// Resting (flat) pulse width for each side.
const int LEFT_START  = 1500;   // GP19 rest
const int RIGHT_START = 1000;   // GP18 rest

// Lift travel. With attach(500, 2500) the servo spans 180 deg over 2000 us,
// so roughly 11 us per degree:
//     1200 us ~= 108 deg  (side panel swung a full 90 deg - too far)
//      915 us ~=  82 deg  - steep enough to block a fall, backed off from
//                            fully vertical so the panel does not overshoot.
const int TRAVEL_MAX   = 726;    // us of travel for a full lift (~65 deg)

// If the zone is STILL danger once the side is at TRAVEL_MAX, push this much
// further - and no further. A last 10 deg of persuasion, not a second lift.
const int DANGER_EXTRA = 111;    // us (~10 deg)

// WARNING angle. The sides rise this far and STAY there while the patient is
// near an edge - braced and ready - without attempting to tilt them back.
// ~675 us is about 61 deg: the angle the sides SIT at while braced. Raised
// so a braced side genuinely resists a roll rather than just leaning in.
const int READY_OFFSET = 675;    // us

// Once the patient is back in the safe zone the sides HOLD their current
// tilt for this long before returning to flat, in case the patient moves
// straight back towards the edge.
const int SETTLE_HOLD_MS  = 10000;  // 10 s

// Movement durations (ms). Longer = smoother and gentler.
const int LIFT_MS  = 1200;  // rest/ready -> full lift: deliberate, not a snap
const int EASE_MS  = 1400;  // moving to the WARNING hold angle
const int LOWER_MS = 1800;  // coming back down: slowest, nothing is urgent

// Which way each servo must move from rest in order to LIFT its side.
// Flip a sign here if a side tilts the wrong way on the real rig.
const int LEFT_DIR  = -1;   // GP19 lifts by decreasing pulse width
const int RIGHT_DIR = +1;   // GP18 lifts by increasing pulse width

// ──────────────────────────────────────────
// State
// ──────────────────────────────────────────
String currentZone    = "SAFE";
unsigned long lastPoll = 0;

// After this many consecutive failed polls we stop trusting the last zone
// and idle at the start positions rather than repeating an old DANGER sweep.
const int MAX_POLL_FAILURES = 3;
int pollFailures = 0;

// Ensures each polled zone triggers its servo action exactly once, so the
// loop stays free to poll again on schedule.
bool actedThisCycle = false;

// ──────────────────────────────────────────
// Helpers — servo moves
// ──────────────────────────────────────────

// Both motors back to rest.
void goToStart() {
  servoLeft.writeMicroseconds(LEFT_START);
  servoRight.writeMicroseconds(RIGHT_START);
}

// Move one servo smoothly from one offset to another.
//
// The servo only refreshes its position every 20 ms (50 Hz), so writing 8 us
// steps every 2 ms just queued ten values per refresh and the motor lurched
// through them — that is what made the tilt look stepwise. Instead, drive it
// on the servo's own 20 ms cadence and ease in/out, so each refresh gets one
// smoothly interpolated position.
void moveTo(Servo& sv, int startUs, int dir, int fromOff, int toOff,
            int durationMs) {
  if (fromOff == toOff) {
    sv.writeMicroseconds(startUs + dir * toOff);
    return;
  }

  const int FRAME_MS = 20;                       // one servo refresh
  int steps = durationMs / FRAME_MS;
  if (steps < 1) steps = 1;

  for (int i = 1; i <= steps; i++) {
    float t = (float)i / (float)steps;           // 0 -> 1
    // Ease in/out (smoothstep): starts gently, accelerates, settles gently,
    // instead of slamming from a standstill to full speed and back.
    float e = t * t * (3.0f - 2.0f * t);
    int off = fromOff + (int)((toOff - fromOff) * e);
    sv.writeMicroseconds(startUs + dir * off);
    delay(FRAME_MS);
  }
  sv.writeMicroseconds(startUs + dir * toOff);
}

// How far each side is currently raised (us above its resting position), so
// moves start from the real position instead of jumping.
int leftOff  = 0;
int rightOff = 0;

// millis() when SAFE was first seen; 0 means "not currently safe".
unsigned long safeSince = 0;

// Re-assert the current position on every pass.
//
// Without this the servo is only ever written to while it is MOVING; once it
// reaches its target nothing refreshes the pulse, and a loaded servo slowly
// sags back. Re-sending the same value each cycle keeps it actively holding
// the tilt for as long as the zone stays unsafe.
void holdPosition() {
  servoLeft.writeMicroseconds(LEFT_START + LEFT_DIR * leftOff);
  servoRight.writeMicroseconds(RIGHT_START + RIGHT_DIR * rightOff);
}

// ── DANGER: lift the side the patient is rolling towards, and HOLD ──
// Raising is one-way: a side only ever goes UP while the patient is unsafe.
// Nothing is lowered until the zone reads SAFE, so a patient shifting around
// near an edge cannot make the servos oscillate.
void liftLeft() {
  safeSince = 0;                     // no longer safe; restart the settle timer

  // Still in danger while already at full lift? Push the last 10 deg, once.
  int target = (leftOff >= TRAVEL_MAX) ? (TRAVEL_MAX + DANGER_EXTRA)
                                       : TRAVEL_MAX;
  if (leftOff < target) {
    moveTo(servoLeft, LEFT_START, LEFT_DIR, leftOff, target, LIFT_MS);
    leftOff = target;
  }
  // The RIGHT side goes FULLY FLAT so the bed leans away from the danger.
  if (rightOff != 0) {
    moveTo(servoRight, RIGHT_START, RIGHT_DIR, rightOff, 0, LOWER_MS);
    rightOff = 0;
  }
  holdPosition();
}

void liftRight() {
  safeSince = 0;

  int target = (rightOff >= TRAVEL_MAX) ? (TRAVEL_MAX + DANGER_EXTRA)
                                        : TRAVEL_MAX;
  if (rightOff < target) {
    moveTo(servoRight, RIGHT_START, RIGHT_DIR, rightOff, target, LIFT_MS);
    rightOff = target;
  }
  if (leftOff != 0) {
    moveTo(servoLeft, LEFT_START, LEFT_DIR, leftOff, 0, LOWER_MS);
    leftOff = 0;
  }
  holdPosition();
}


// ── WARNING: brace both sides and STAY braced ──
// The patient is still on the bed, so a full lift would achieve nothing. Both
// sides rise to READY_OFFSET and hold.
//
// Crucially this NEVER lowers a side. The zone flickers between DANGER and
// WARNING while a patient shifts around near an edge, and dropping back to the
// ready angle on every WARNING made the servos pump up and down. A side that is
// already higher than READY_OFFSET simply stays where it is; nothing comes down
// until the patient is genuinely SAFE again.
// Brace ONE side at the ready angle; the other goes flat.
//
// The zone now names the side (WARNING_LEFT / WARNING_RIGHT), so only the
// threatened edge is raised. Raising both meant the opposite motor moved for
// no reason, and the bed rose evenly instead of leaning away from the edge.
void braceSide(bool left) {
  safeSince = 0;

  int wantLeft  = left ? READY_OFFSET : 0;
  int wantRight = left ? 0 : READY_OFFSET;

  if (leftOff != wantLeft) {
    moveTo(servoLeft, LEFT_START, LEFT_DIR, leftOff, wantLeft,
           leftOff < wantLeft ? EASE_MS : LOWER_MS);
    leftOff = wantLeft;
  }
  if (rightOff != wantRight) {
    moveTo(servoRight, RIGHT_START, RIGHT_DIR, rightOff, wantRight,
           rightOff < wantRight ? EASE_MS : LOWER_MS);
    rightOff = wantRight;
  }
  holdPosition();
}


// ── SAFE again: hold the current tilt, then return to flat ──
// The sides stay exactly where they are for SETTLE_HOLD_MS after the patient
// first reads SAFE, in case they move straight back towards the edge. Only
// once that has passed do both sides ease down to flat.
//
// safeSince is cleared by liftLeft/liftRight/braceSide, so any DANGER or
// WARNING in the meantime restarts the countdown - the sides never drop while
// the zone is still oscillating.
void lowerAll() {
  if (leftOff == 0 && rightOff == 0) {
    safeSince = 0;
    return;                                  // already flat, nothing to do
  }

  if (safeSince == 0) safeSince = millis();  // first SAFE reading: start timer

  if (millis() - safeSince < (unsigned long)SETTLE_HOLD_MS) {
    holdPosition();                          // keep the tilt while waiting
    return;
  }

  if (leftOff != 0) {
    moveTo(servoLeft, LEFT_START, LEFT_DIR, leftOff, 0, LOWER_MS);
    leftOff = 0;
  }
  if (rightOff != 0) {
    moveTo(servoRight, RIGHT_START, RIGHT_DIR, rightOff, 0, LOWER_MS);
    rightOff = 0;
  }
  safeSince = 0;
}

// ──────────────────────────────────────────
// WiFi connect helper
// ──────────────────────────────────────────
void connectWiFi() {
  Serial.print("  Connecting to WiFi: ");
  Serial.print(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("  Connected! ESP32 IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("  WiFi connection FAILED — will retry in loop.");
  }
}

// ──────────────────────────────────────────
// Minimal JSON field readers
//
// The reply from app.py has a fixed, known shape, so these pull out the two
// fields this sketch needs. Keeping it dependency-free means the sketch
// compiles on a bare Arduino IDE with only ESP32Servo installed.
// ──────────────────────────────────────────

// Value of a string field, e.g. jsonString(body, "status") -> "DANGER_LEFT".
// Returns "" when the key is absent.
String jsonString(const String& body, const char* key) {
  String needle = String("\"") + key + "\"";
  int k = body.indexOf(needle);
  if (k < 0) return "";

  int colon = body.indexOf(':', k + needle.length());
  if (colon < 0) return "";

  int firstQuote = body.indexOf('"', colon + 1);
  if (firstQuote < 0) return "";

  int lastQuote = body.indexOf('"', firstQuote + 1);
  if (lastQuote < 0) return "";

  return body.substring(firstQuote + 1, lastQuote);
}

// True when a boolean field is literally true, e.g. "stale":true
bool jsonIsTrue(const String& body, const char* key) {
  String needle = String("\"") + key + "\"";
  int k = body.indexOf(needle);
  if (k < 0) return false;

  int colon = body.indexOf(':', k + needle.length());
  if (colon < 0) return false;

  // Skip whitespace after the colon, then look for "true".
  int p = colon + 1;
  while (p < (int)body.length() && (body[p] == ' ' || body[p] == '\t')) p++;
  return body.startsWith("true", p);
}

// ──────────────────────────────────────────
// Poll Flask server for the current zone
// ──────────────────────────────────────────
String fetchZone() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("  [WiFi] Not connected — skipping poll.");
    return currentZone;   // keep last known zone
  }

  // Build URL dynamically so SERVER_IP is used
  char url[64];
  snprintf(url, sizeof(url), "http://%s:%d/status", SERVER_IP, SERVER_PORT);

  HTTPClient http;
  http.begin(url);
  http.setTimeout(1500);   // 1.5 s timeout so we never block the loop too long

  int code = http.GET();
  String zone = currentZone;   // default: keep last

  if (code == HTTP_CODE_OK) {
    String payload = http.getString();
    // app.py returns e.g.
    // {"status":"DANGER_LEFT","confidence":0.87,"cx":210,"stale":false,...}
    // Only two fields matter here, so they are pulled out directly rather
    // than pulling in a whole JSON library for a fixed, known payload.
    String parsed = jsonString(payload, "status");

    if (parsed.length() > 0) {
      zone = parsed;
      zone.trim();
      zone.toUpperCase();
      pollFailures = 0;

      // The server flags a stale reading when img_process.py has stopped
      // refreshing status.json — treat it as "no command".
      if (jsonIsTrue(payload, "stale")) {
        zone = "STALE";
      }
    } else {
      Serial.println("  [HTTP] could not read \"status\" from reply");
      pollFailures++;
    }
  } else {
    Serial.print("  [HTTP] Error code: ");
    Serial.println(code);
    pollFailures++;
  }

  http.end();

  if (pollFailures >= MAX_POLL_FAILURES) {
    Serial.println("  [HTTP] Server unreachable — idling at start positions.");
    zone = "STALE";
  }

  return zone;
}

// ──────────────────────────────────────────
// setup
// ──────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("========================================");
  Serial.println("  Sleeping Monitor — Motor Controller");
  Serial.println("========================================");

  // ── Servo init — identical to original ──
  servoLeft.setPeriodHertz(50);
  servoRight.setPeriodHertz(50);

  servoLeft.attach(PIN_LEFT,   500, 2500);   // GP19 -> LEFT  side
  servoRight.attach(PIN_RIGHT, 500, 2500);   // GP18 -> RIGHT side

  // Snap to the flat resting position on power-up.
  servoLeft.writeMicroseconds(LEFT_START);
  servoRight.writeMicroseconds(RIGHT_START);
  leftOff = rightOff = 0;

  delay(3000);                              // let the motors physically arrive
  Serial.println("  Servos at start positions.");

  // ── WiFi ──
  connectWiFi();

  lastPoll = millis();
  Serial.println("  Polling server every 2 s…");
}

// ──────────────────────────────────────────
// loop
// ──────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  // ── Reconnect if WiFi dropped ──
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("  [WiFi] Disconnected — reconnecting…");
    WiFi.disconnect();
    connectWiFi();
  }

  // ── Poll every POLL_INTERVAL_MS ──
  if (now - lastPoll >= POLL_INTERVAL_MS) {
    lastPoll = now;
    actedThisCycle = false;   // allow one action for this new reading

    String newZone = fetchZone();

    if (newZone != currentZone) {
      Serial.print("  Zone changed: ");
      Serial.print(currentZone);
      Serial.print(" -> ");
      Serial.println(newZone);
      currentZone = newZone;
      // React to the change on this very pass.
      actedThisCycle = false;
    } else {
      Serial.print("  Zone: ");
      Serial.println(currentZone);
    }
  }

  // ── Act on current zone ──
  // Each action runs once per poll cycle so a zone change is picked up
  // within POLL_INTERVAL_MS instead of being buried under a long sweep.
  if (currentZone == "DANGER_LEFT") {
    if (!actedThisCycle) {
      actedThisCycle = true;
      Serial.println("  [ACTION] DANGER_LEFT - lifting LEFT edge");
      liftLeft();
    } else {
      holdPosition();
    }

  } else if (currentZone == "DANGER_RIGHT") {
    if (!actedThisCycle) {
      actedThisCycle = true;
      Serial.println("  [ACTION] DANGER_RIGHT - lifting RIGHT edge");
      liftRight();
    } else {
      holdPosition();
    }

  } else if (currentZone == "WARNING_LEFT" || currentZone == "WARNING_RIGHT") {
    // No full lift here - the patient is still on the bed. Only the
    // threatened side braces, and it stays there while WARNING lasts.
    if (!actedThisCycle) {
      actedThisCycle = true;
      bool left = (currentZone == "WARNING_LEFT");
      Serial.print("  [ACTION] WARNING - bracing ");
      Serial.print(left ? "LEFT" : "RIGHT");
      Serial.println(" side");
      braceSide(left);
    } else {
      // actedThisCycle stops the MOVE repeating, but the servo still needs a
      // pulse every cycle or it sags out of position under load.
      holdPosition();
    }

  } else {
    // SAFE / EMPTY / NOT_FOUND / STALE / unknown -> lower both sides
    if (!actedThisCycle) {
      actedThisCycle = true;
      lowerAll();
    }
    delay(20);   // small yield so the WiFi stack can breathe
  }
}
