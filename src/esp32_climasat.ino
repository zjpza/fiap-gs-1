/*
  ClimaSat — Step 5: Firmware ESP32
  ===================================
  Sensores:
    - DHT22  → temperatura (°C) e umidade relativa (%)
    - BMP280 → pressão atmosférica (hPa) e altitude estimada (m)

  Saídas:
    - Serial USB  : JSON a cada 5s  →  lido pelo climasat_serial_reader.py
    - Wi-Fi HTTP  : POST JSON para o servidor Python (opcional)
    - LED interno : pisca a cada leitura bem-sucedida

  Bibliotecas necessárias (instale pelo Library Manager da Arduino IDE):
    - DHT sensor library  (Adafruit)
    - Adafruit BMP280
    - Adafruit Unified Sensor
    - ArduinoJson

  Pinagem:
    DHT22  → DATA: GPIO 4  |  VCC: 3.3V  |  GND: GND  (resistor 10kΩ DATA→VCC)
    BMP280 → SDA: GPIO 21  |  SCL: GPIO 22  |  VCC: 3.3V  |  GND: GND
    LED    → GPIO 2 (LED interno do ESP32)
*/

#include <Arduino.h>
#include <DHT.h>
#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <ArduinoJson.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ── Configurações ─────────────────────────────
#define DHT_PIN      4
#define DHT_TYPE     DHT22
#define LED_PIN      2
#define INTERVALO_MS 5000      // leitura a cada 5 segundos

// Wi-Fi (deixe vazio para usar apenas Serial)
const char* WIFI_SSID     = "";
const char* WIFI_PASSWORD = "";

// Endereço do servidor Python (HTTP POST)
// Ex: "http://192.168.1.100:8765/sensor"
const char* SERVER_URL = "";

// ── Objetos ───────────────────────────────────
DHT            dht(DHT_PIN, DHT_TYPE);
Adafruit_BMP280 bmp;

bool bmp_ok     = false;
bool wifi_ok    = false;
int  leitura_id = 0;

// ── Setup ─────────────────────────────────────
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);

  // Inicializa DHT22
  dht.begin();
  delay(2000);  // DHT22 precisa de 2s para estabilizar

  // Inicializa BMP280 (endereço I2C padrão 0x76 ou 0x77)
  if (bmp.begin(0x76) || bmp.begin(0x77)) {
    bmp_ok = true;
    bmp.setSampling(
      Adafruit_BMP280::MODE_NORMAL,
      Adafruit_BMP280::SAMPLING_X2,
      Adafruit_BMP280::SAMPLING_X16,
      Adafruit_BMP280::FILTER_X16,
      Adafruit_BMP280::STANDBY_MS_500
    );
  }

  // Conecta Wi-Fi se configurado
  if (strlen(WIFI_SSID) > 0) {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int tentativas = 0;
    while (WiFi.status() != WL_CONNECTED && tentativas < 20) {
      delay(500);
      tentativas++;
    }
    wifi_ok = (WiFi.status() == WL_CONNECTED);
  }

  // Mensagem de inicialização
  Serial.println("{\"status\":\"CLIMASAT_READY\","
                 "\"sensores\":{\"dht22\":true,"
                 "\"bmp280\":" + String(bmp_ok ? "true" : "false") + ","
                 "\"wifi\":"   + String(wifi_ok ? "true" : "false") + "}}");
}

// ── Leitura dos sensores ──────────────────────
bool lerSensores(float &temp, float &umid, float &pressao, float &altitude) {
  temp     = dht.readTemperature();
  umid     = dht.readHumidity();
  pressao  = bmp_ok ? bmp.readPressure() / 100.0F : -1;
  altitude = bmp_ok ? bmp.readAltitude(1013.25)   : -1;

  // Valida leituras do DHT22
  if (isnan(temp) || isnan(umid)) return false;
  if (temp < -40 || temp > 80)   return false;
  if (umid < 0   || umid > 100)  return false;
  return true;
}

// ── Monta JSON de leitura ────────────────────
String montarJSON(float temp, float umid, float pressao, float altitude) {
  StaticJsonDocument<256> doc;

  doc["id"]        = leitura_id++;
  doc["timestamp"] = millis() / 1000;   // segundos desde boot
  doc["temp"]      = round(temp  * 10) / 10.0;
  doc["umidade"]   = round(umid  * 10) / 10.0;
  doc["pressao"]   = bmp_ok ? round(pressao  * 10) / 10.0 : nullptr;
  doc["altitude"]  = bmp_ok ? round(altitude * 10) / 10.0 : nullptr;
  doc["wifi"]      = wifi_ok;
  doc["status"]    = "OK";

  String output;
  serializeJson(doc, output);
  return output;
}

// ── Envia via HTTP (se Wi-Fi configurado) ─────
void enviarHTTP(const String& payload) {
  if (!wifi_ok || strlen(SERVER_URL) == 0) return;
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(payload);
  http.end();
}

// ── Pisca LED de status ───────────────────────
void piscarLED(int vezes, int ms) {
  for (int i = 0; i < vezes; i++) {
    digitalWrite(LED_PIN, HIGH); delay(ms);
    digitalWrite(LED_PIN, LOW);  delay(ms);
  }
}

// ── Loop principal ────────────────────────────
void loop() {
  float temp, umid, pressao, altitude;

  if (lerSensores(temp, umid, pressao, altitude)) {
    String json = montarJSON(temp, umid, pressao, altitude);

    // Envia pelo Serial (sempre)
    Serial.println(json);

    // Envia pelo Wi-Fi (se configurado)
    enviarHTTP(json);

    // LED: 1 piscada = leitura OK
    piscarLED(1, 100);

  } else {
    // Erro de leitura
    Serial.println("{\"status\":\"ERRO\",\"msg\":\"Falha no DHT22\"}");
    piscarLED(3, 200);   // 3 piscadas = erro
  }

  delay(INTERVALO_MS);
}
