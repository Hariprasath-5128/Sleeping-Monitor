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
Servo servo1;
Servo servo2;

const int SERVO1_START = 1000;   // GP18 start position
const int SERVO2_START = 1500;   // GP19 start position
const int TRAVEL_MAX   = 800;    // µs travel range

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

// Return both servos to their individual start positions
void goToStart() {
  servo1.writeMicroseconds(SERVO1_START);
  servo2.writeMicroseconds(SERVO2_START);
}

// Full sweep (same as original loop behaviour) — used on DANGER
void fullSweep() {
  // Forward
  for (int offset = 0; offset <= TRAVEL_MAX; offset += 4) {
    servo1.writeMicroseconds(SERVO1_START + offset);
    servo2.writeMicroseconds(SERVO2_START - offset);
    delay(2);
  }
  // Reverse — return to start
  for (int offset = TRAVEL_MAX; offset >= 0; offset -= 4) {
    servo1.writeMicroseconds(SERVO1_START + offset);
    servo2.writeMicroseconds(SERVO2_START - offset);
    delay(2);
  }
}

// Gentle nudge (~30 % of travel) — used on WARNING
void gentleNudge() {
  int nudge = TRAVEL_MAX * 30 / 100;  // 30 % of TRAVEL_MAX
  // Forward nudge
  for (int offset = 0; offset <= nudge; offset += 4) {
    servo1.writeMicroseconds(SERVO1_START + offset);
    servo2.writeMicroseconds(SERVO2_START - offset);
    delay(2);
  }
  delay(200);
  // Return to start
  for (int offset = nudge; offset >= 0; offset -= 4) {
    servo1.writeMicroseconds(SERVO1_START + offset);
    servo2.writeMicroseconds(SERVO2_START - offset);
    delay(2);
  }
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
  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);

  servo1.attach(18, 500, 2500);   // GP18
  servo2.attach(19, 500, 2500);   // GP19

  // Snap to individual starting positions on power-up
  servo1.writeMicroseconds(SERVO1_START);
  servo2.writeMicroseconds(SERVO2_START);

  // Let motors physically reach their start positions
  delay(3000);
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
  if (currentZone == "DANGER_LEFT" || currentZone == "DANGER_RIGHT") {
    if (!actedThisCycle) {
      actedThisCycle = true;
      Serial.println("  [ACTION] DANGER — full sweep");
      fullSweep();
    }

  } else if (currentZone == "WARNING") {
    if (!actedThisCycle) {
      actedThisCycle = true;
      Serial.println("  [ACTION] WARNING — gentle nudge");
      gentleNudge();
    }

  } else {
    // SAFE / EMPTY / NOT_FOUND / STALE / unknown -> hold at start positions
    if (!actedThisCycle) {
      actedThisCycle = true;
      goToStart();
    }
    delay(20);   // small yield so the WiFi stack can breathe
  }
}
