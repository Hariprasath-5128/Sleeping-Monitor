#include <WiFi.h>
#include <HTTPClient.h>

// Put the ESP32 and your Python server machine on the same WiFi network.
const char* WIFI_SSID = "Hari";
const char* WIFI_PASSWORD = "28022007";

// Replace 192.168.1.10 with the IP address of the PC running web_server.py.
// Keep the port as 5050 unless you change PORT in web_server.py.
const char* RESULT_URL = "http://192.168.1.4:5050/result";

const unsigned long POLL_INTERVAL_MS = 2000;
unsigned long lastPollMs = 0;

void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi connected");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

void fetchMonitorResult() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected. Reconnecting...");
    connectWiFi();
  }

  HTTPClient http;
  http.setTimeout(3000);
  http.begin(RESULT_URL);

  int statusCode = http.GET();
  if (statusCode > 0) {
    Serial.print("HTTP ");
    Serial.println(statusCode);

    String payload = http.getString();
    Serial.println("Sleeping Monitor Result:");
    Serial.println(payload);
    Serial.println();
  } else {
    Serial.print("HTTP request failed: ");
    Serial.println(http.errorToString(statusCode));
  }

  http.end();
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  connectWiFi();
}

void loop() {
  unsigned long now = millis();
  if (now - lastPollMs >= POLL_INTERVAL_MS) {
    lastPollMs = now;
    fetchMonitorResult();
  }
}
