/*
 * ============================================================
 *  ESP32 RGB WebSocket Server  (Core 3.x)
 * ============================================================
 *  RGB Pins: R=GPIO2, G=GPIO4, B=GPIO5, GND=GND
 *  WebSocket Port: 81
 * ============================================================
 */

#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>

// ============ CHANGE THESE ============
const char* ssid     = "Estrera2.4";
const char* password = "S3RAFAM2";
// ======================================

#define PIN_R 2
#define PIN_G 4
#define PIN_B 5

WebServer server(80);
WebSocketsServer webSocket(81);

void setup() {
  Serial.begin(115200);
  delay(500);

  ledcAttach(PIN_R, 5000, 8);
  ledcAttach(PIN_G, 5000, 8);
  ledcAttach(PIN_B, 5000, 8);
  ledcWrite(PIN_R, 0);
  ledcWrite(PIN_G, 0);
  ledcWrite(PIN_B, 0);

  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  webSocket.begin();
  webSocket.onEvent(webSocketEvent);

  server.on("/", []() {
    server.send(200, "text/plain", "ESP32 RGB WebSocket running");
  });
  server.begin();

  Serial.println("Ready.");
}

void loop() {
  webSocket.loop();
  server.handleClient();
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t *payload, size_t length) {
  if (type == WStype_TEXT) {
    String msg = String((char*)payload);
    msg.trim();
    int c1 = msg.indexOf(',');
    int c2 = msg.indexOf(',', c1 + 1);
    if (c1 > 0 && c2 > 0) {
      int r = constrain(msg.substring(0, c1).toInt(), 0, 255);
      int g = constrain(msg.substring(c1 + 1, c2).toInt(), 0, 255);
      int b = constrain(msg.substring(c2 + 1).toInt(), 0, 255);
      ledcWrite(PIN_R, r);
      ledcWrite(PIN_G, g);
      ledcWrite(PIN_B, b);
      Serial.printf("RGB: %d,%d,%d\n", r, g, b);
    }
  }
}
