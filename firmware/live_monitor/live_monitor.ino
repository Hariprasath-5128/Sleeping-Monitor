// ==========================================
// Sleeping Monitor — Live Stream (ESP32-CAM)
// AI-Thinker ESP32-CAM board. Streams MJPEG so
// img_process.py can consume it as a video source.
//
//   http://<CAM_IP>/           - landing page
//   http://<CAM_IP>:81/stream  - dedicated multipart MJPEG stream
//   http://<CAM_IP>/capture    - single JPEG frame
//
// Board:  "AI Thinker ESP32-CAM"
// PSRAM:  Enabled
// Upload: 115200 baud, GPIO0 -> GND while flashing
// ==========================================

#include <WiFi.h>
#include <esp_camera.h>
#include <esp_http_server.h>
#include <esp_arduino_version.h>   // ESP_ARDUINO_VERSION_MAJOR (LEDC API choice)
#include <esp_wifi.h>              // esp_wifi_set_ps() - kill WiFi power save

// ──────────────────────────────────────────
// WiFi credentials  ← must match the motor ESP32 and your PC
// ──────────────────────────────────────────
const char* WIFI_SSID     = "Hariprasath LAPTOP";
const char* WIFI_PASSWORD = "123456789";

// Optional: pin a static IP so img_process.py never needs editing.
// Leave USE_STATIC_IP false to let the router assign one (printed on Serial).
#define USE_STATIC_IP false
IPAddress staticIP(192, 168, 1, 50);
IPAddress gateway (192, 168, 1, 1);
IPAddress subnet  (255, 255, 255, 0);

// ──────────────────────────────────────────
// Camera pin map — AI-Thinker ESP32-CAM
// ──────────────────────────────────────────
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define LED_FLASH_GPIO     4   // on-board white flash LED
#define LED_FLASH_CHANNEL  2   // LEDC channel (0/1 are used by the camera)

// Flash LED brightness, 0-255. 0 = off.
// The LED draws real current and runs hot, which on this board has coincided
// with the WiFi stalls, so it stays off by default. Raise it only if the bed
// is genuinely too dark for the camera.
#define FLASH_BRIGHTNESS   0

// Dedicated stream port, matching the reference project's server layout.
#define STREAM_PORT 81
#define PART_BOUNDARY "sleepingmonitorframe"
#define STREAM_CONTENT_TYPE "multipart/x-mixed-replace;boundary=" PART_BOUNDARY
#define STREAM_BOUNDARY "\r\n--" PART_BOUNDARY "\r\n"
#define STREAM_PART "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n"

httpd_handle_t controlServer = NULL;
httpd_handle_t streamServer = NULL;

// ──────────────────────────────────────────
// Camera init
// ──────────────────────────────────────────
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  // 10 MHz rather than the usual 20 MHz: many AI-Thinker boards produce
  // corrupt JPEGs at 20 MHz ("huffman table decode error", "overread"),
  // especially on marginal power. Half the clock is far more reliable and
  // still comfortably fast enough for QVGA.
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Tuned for LATENCY, not picture quality.
  //
  // QQVGA (160x120) with heavy JPEG compression cuts a frame from ~4.6 KB to
  // roughly 1-1.5 KB. Less to encode and less to push through a struggling
  // radio means the camera answers sooner, which is the whole problem here.
  // YOLO still works at this size for a body-sized object on a bed; raise
  // these back up (QVGA / quality 12) once the link is healthy.
  if (psramFound()) {
    config.frame_size   = FRAMESIZE_QQVGA;  // 160x120
    config.jpeg_quality = 35;               // higher number = smaller frame
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size   = FRAMESIZE_QQVGA;
    config.jpeg_quality = 40;
    config.fb_count     = 1;
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("  [CAM] init failed: 0x%x\n", err);
    return false;
  }

  // The bed is viewed from above; flip/mirror so the operator sees it the
  // same way round as the IP-webcam feed this replaces.
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, 1);
    s->set_hmirror(s, 0);
    s->set_brightness(s, 1);
    s->set_saturation(s, 0);
  }
  return true;
}

// ──────────────────────────────────────────
// WiFi connect helper
// ──────────────────────────────────────────
void connectWiFi() {
  Serial.print("  Connecting to WiFi: ");
  Serial.print(WIFI_SSID);

  WiFi.mode(WIFI_STA);
#if USE_STATIC_IP
  WiFi.config(staticIP, gateway, subnet);
#endif
  WiFi.setSleep(false);          // keep latency low for streaming
  // Max transmit power: the stock setting is conservative and, on a link with
  // any distance or interference, causes the retries that show up as multi-
  // second response times rather than as outright packet loss.
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);        // do not rewrite flash on every connect
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    // Re-apply AFTER the association completes. Setting these before
    // WiFi.begin() is not enough — the driver reconfigures the radio when it
    // associates, which re-enables power save. Power save is what makes the
    // camera answer in 30 ms one moment and 1000 ms the next.
    WiFi.setSleep(false);
    WiFi.setTxPower(WIFI_POWER_19_5dBm);
    esp_wifi_set_ps(WIFI_PS_NONE);   // the one that actually sticks

    Serial.print("  Connected! Camera IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("  Stream URL  -> http://");
    Serial.print(WiFi.localIP());
    Serial.print(":");
    Serial.print(STREAM_PORT);
    Serial.println("/stream");
    Serial.println("  Set CAM_IP in img_process.py; it uses stream port 81.");
  } else {
    Serial.println("  WiFi FAILED - restarting in 5 s...");
    delay(5000);
    ESP.restart();
  }
}

// ──────────────────────────────────────────
// Handlers
// ──────────────────────────────────────────
esp_err_t handleRoot(httpd_req_t* req) {
  String ip = WiFi.localIP().toString();
  String html =
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>Sleeping Monitor Camera</title>"
    "<style>body{background:#111;color:#eee;font-family:sans-serif;"
    "text-align:center;padding:16px}"
    "img{max-width:100%;border:2px solid #0cf;border-radius:6px}"
    "code{background:#222;padding:2px 6px;border-radius:4px;color:#0cf}"
    "</style></head><body>"
    "<h2>Sleeping Monitor &mdash; Live Camera</h2>"
    "<p>Feed this URL to img_process.py: <code>http://" + ip + ":" +
    String(STREAM_PORT) + "/stream</code></p>"
    "<img src='http://" + ip + ":" + String(STREAM_PORT) + "/stream'></body></html>";
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, html.c_str(), html.length());
}

esp_err_t handleCapture(httpd_req_t* req) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    return httpd_resp_send_500(req);
  }
  httpd_resp_set_type(req, "image/jpeg");
  esp_err_t result = httpd_resp_send(req, reinterpret_cast<const char*>(fb->buf), fb->len);
  esp_camera_fb_return(fb);
  return result;
}

// Adapted from the reference project's esp_http_server stream handler.
// A dedicated HTTP server prevents the page/capture endpoints from blocking MJPEG.
esp_err_t handleStream(httpd_req_t* req) {
  esp_err_t result = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (result != ESP_OK) return result;

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
  Serial.println("  [STREAM] client connected");

  int grabFailures = 0;

  while (result == ESP_OK) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      // A single dropped grab is normal under load. Tearing the whole stream
      // down for one is what made the client reconnect in a loop.
      if (++grabFailures >= 10) {
        Serial.println("  [STREAM] camera stopped delivering frames");
        result = ESP_FAIL;
        break;
      }
      delay(20);
      continue;
    }
    grabFailures = 0;

    char part[96];
    size_t partLength = snprintf(part, sizeof(part), STREAM_PART, fb->len);
    result = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (result == ESP_OK) result = httpd_resp_send_chunk(req, part, partLength);
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(req, reinterpret_cast<const char*>(fb->buf), fb->len);
    }

    esp_camera_fb_return(fb);

    // Hand the CPU to the WiFi/TCP task without burning a fixed 5 ms per
    // frame: at 12 fps that delay alone was capping the rate. yield() lets
    // the stack drain and returns as soon as it is done.
    yield();
  }

  Serial.println("  [STREAM] client disconnected");
  return result;
}

void startCameraServers() {
  httpd_config_t controlConfig = HTTPD_DEFAULT_CONFIG();
  controlConfig.max_uri_handlers = 2;

  httpd_uri_t rootUri = {};
  rootUri.uri = "/";
  rootUri.method = HTTP_GET;
  rootUri.handler = handleRoot;

  httpd_uri_t captureUri = {};
  captureUri.uri = "/capture";
  captureUri.method = HTTP_GET;
  captureUri.handler = handleCapture;

  esp_err_t err = httpd_start(&controlServer, &controlConfig);
  if (err == ESP_OK) {
    httpd_register_uri_handler(controlServer, &rootUri);
    httpd_register_uri_handler(controlServer, &captureUri);
    Serial.println("  Control server started on port 80.");
  } else {
    Serial.printf("  [HTTP] control server failed to start: 0x%x\n", err);
  }

  httpd_config_t streamConfig = HTTPD_DEFAULT_CONFIG();
  streamConfig.server_port = STREAM_PORT;
  // Must differ from the control server's ctrl_port, or the second
  // httpd_start() fails and port 81 resets every connection.
  streamConfig.ctrl_port = controlConfig.ctrl_port + 1;
  streamConfig.max_uri_handlers = 1;
  // MJPEG is ONE long-lived request. Allowing a second socket while
  // lru_purge_enable is on makes the server evict the oldest connection -
  // which is the live stream - so the client sees "stream ends prematurely",
  // reconnects, gets purged again, and loops forever. One socket, no purging.
  streamConfig.max_open_sockets = 1;
  streamConfig.lru_purge_enable = false;
  streamConfig.stack_size = 8192;
  // Do not let the server time out a stream that is deliberately endless.
  streamConfig.recv_wait_timeout = 10;
  streamConfig.send_wait_timeout = 10;

  httpd_uri_t streamUri = {};
  streamUri.uri = "/stream";
  streamUri.method = HTTP_GET;
  streamUri.handler = handleStream;

  err = httpd_start(&streamServer, &streamConfig);
  if (err == ESP_OK) {
    httpd_register_uri_handler(streamServer, &streamUri);
    Serial.printf("  Stream server started on port %d.\n", STREAM_PORT);
  } else {
    Serial.printf("  [HTTP] STREAM SERVER FAILED TO START: 0x%x\n", err);
    Serial.println("  -> port 81 will refuse/reset connections.");
  }
}

// ──────────────────────────────────────────
// setup
// ──────────────────────────────────────────
void setup() {
  // Run the CPU flat out. Anything less throttles both JPEG encoding and the
  // WiFi stack, which is what turns a 5 ms request into a multi-second one.
  setCpuFrequencyMhz(240);

  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.println();
  Serial.println("========================================");
  Serial.println("  Sleeping Monitor - ESP32-CAM Streamer");
  Serial.println("========================================");

  // Flash LED on continuously to light the bed.
  // Driven by PWM rather than digitalWrite(HIGH): at full brightness this LED
  // pulls a lot of current and runs hot, which on a marginal supply is exactly
  // what browns the board out mid-stream. FLASH_BRIGHTNESS is the dial.
  //
  // ESP32 core 3.x replaced ledcSetup()/ledcAttachPin() with a single
  // ledcAttach(pin, freq, resolution), so pick the API the installed core has.
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(LED_FLASH_GPIO, 5000, 8);          // 5 kHz, 8-bit
  ledcWrite(LED_FLASH_GPIO, FLASH_BRIGHTNESS);  // 3.x addresses the PIN
#else
  ledcSetup(LED_FLASH_CHANNEL, 5000, 8);
  ledcAttachPin(LED_FLASH_GPIO, LED_FLASH_CHANNEL);
  ledcWrite(LED_FLASH_CHANNEL, FLASH_BRIGHTNESS);
#endif

  if (!initCamera()) {
    Serial.println("  Camera init failed - restarting in 5 s...");
    delay(5000);
    ESP.restart();
  }
  Serial.println("  Camera ready.");
  Serial.print("  PSRAM detected: ");
  Serial.println(psramFound() ? "yes" : "no");

  connectWiFi();

  startCameraServers();
}

// ──────────────────────────────────────────
// loop
// ──────────────────────────────────────────
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("  [WiFi] Disconnected - reconnecting...");
    connectWiFi();
  }
}
