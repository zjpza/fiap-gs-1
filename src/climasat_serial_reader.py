"""
ClimaSat — Step 5: Leitor Serial do ESP32
==========================================
Lê dados JSON do ESP32 via USB/Serial, salva em CSV
e serve um endpoint HTTP para o dashboard Streamlit.

Uso:
    python climasat_serial_reader.py --porta COM3   (Windows)
    python climasat_serial_reader.py --porta /dev/ttyUSB0  (Linux/Mac)

O arquivo  esp32_leituras.json  é atualizado a cada leitura
e lido pelo dashboard em tempo real.
"""

import serial
import serial.tools.list_ports
import json
import csv
import time
import os
import argparse
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

# ──────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────
BAUD_RATE    = 115200
OUTPUT_CSV   = "climasat_output/esp32_leituras.csv"
OUTPUT_JSON  = "climasat_output/esp32_ultimo.json"
HTTP_PORT    = 8765
MAX_HISTORICO = 1440   # máx. 1440 leituras (24h a 1/min)

os.makedirs("climasat_output", exist_ok=True)

ultima_leitura = {}
historico = []
lock = threading.Lock()


# ──────────────────────────────────────────────
# DETECTAR PORTA ESP32 AUTOMATICAMENTE
# ──────────────────────────────────────────────
def detectar_porta():
    """Tenta encontrar o ESP32 automaticamente."""
    portas = serial.tools.list_ports.comports()
    candidatos = []
    for p in portas:
        desc = (p.description or "").lower()
        if any(k in desc for k in ["cp210", "ch340", "uart", "usb serial", "esp"]):
            candidatos.append(p.device)

    if candidatos:
        print(f"[Auto-detect] ESP32 encontrado em: {candidatos[0]}")
        return candidatos[0]

    # Fallback: lista todas as portas
    if portas:
        print("[Auto-detect] Portas disponíveis:")
        for p in portas:
            print(f"  {p.device} — {p.description}")
        return portas[0].device

    return None


# ──────────────────────────────────────────────
# SALVAR LEITURAS
# ──────────────────────────────────────────────
def salvar_csv(dados):
    """Salva leitura no CSV histórico."""
    campos = ["timestamp_iso","id","temp","umidade","pressao","altitude","wifi","status"]
    existe = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        if not existe:
            writer.writeheader()
        writer.writerow(dados)


def salvar_json(dados):
    """Salva última leitura em JSON (lido pelo dashboard)."""
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# SERVIDOR HTTP SIMPLES
# ──────────────────────────────────────────────
class SensorHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass   # silencia logs do servidor

    def do_GET(self):
        if self.path in ["/", "/sensor", "/ultimo"]:
            with lock:
                dados = ultima_leitura.copy()
            body = json.dumps(dados, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/historico":
            with lock:
                hist = historico[-100:]   # últimos 100
            body = json.dumps(hist, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        # Recebe dados do ESP32 via Wi-Fi
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length).decode("utf-8")
        try:
            dados = json.loads(body)
            processar_leitura(dados)
            self.send_response(200)
        except Exception as e:
            self.send_response(400)
        self.end_headers()


def iniciar_servidor():
    servidor = HTTPServer(("0.0.0.0", HTTP_PORT), SensorHandler)
    print(f"[HTTP] Servidor rodando em http://localhost:{HTTP_PORT}/sensor")
    servidor.serve_forever()


# ──────────────────────────────────────────────
# PROCESSAR LEITURA
# ──────────────────────────────────────────────
def processar_leitura(dados):
    global ultima_leitura, historico

    if dados.get("status") != "OK":
        print(f"[AVISO] {dados}")
        return

    # Adiciona timestamp ISO
    dados["timestamp_iso"] = datetime.now().isoformat()

    with lock:
        ultima_leitura = dados.copy()
        historico.append(dados.copy())
        if len(historico) > MAX_HISTORICO:
            historico.pop(0)

    salvar_csv(dados)
    salvar_json(dados)

    # Log formatado no terminal
    t = dados.get("temp",   "?")
    u = dados.get("umidade","?")
    p = dados.get("pressao","?")
    a = dados.get("altitude","?")
    print(f"[{dados['timestamp_iso'][11:19]}] "
          f"Temp: {t}°C  Umid: {u}%  "
          f"Pressão: {p} hPa  Alt: {a}m")


# ──────────────────────────────────────────────
# MODO SIMULAÇÃO (sem ESP32 físico)
# ──────────────────────────────────────────────
def simular_esp32():
    """Gera leituras simuladas para testes sem hardware."""
    import numpy as np
    print("[SIMULAÇÃO] Gerando leituras simuladas do ESP32...")
    base_t = 22.0
    base_u = 75.0
    base_p = 1013.0
    i = 0
    while True:
        hora = datetime.now().hour
        # Simula variação diurna de temperatura
        t = base_t + 4 * np.sin(np.pi * hora / 12) + np.random.normal(0, .3)
        u = base_u - 2 * np.sin(np.pi * hora / 12) + np.random.normal(0, 1)
        p = base_p + np.random.normal(0, .5)
        a = 44330 * (1 - (p / 1013.25) ** 0.1903)
        dados = {
            "id":        i,
            "timestamp": i * 5,
            "temp":      round(t, 1),
            "umidade":   round(u, 1),
            "pressao":   round(p, 1),
            "altitude":  round(a, 1),
            "wifi":      False,
            "status":    "OK",
        }
        processar_leitura(dados)
        i += 1
        time.sleep(5)


# ──────────────────────────────────────────────
# LOOP PRINCIPAL — LEITURA SERIAL
# ──────────────────────────────────────────────
def ler_serial(porta):
    print(f"[Serial] Conectando em {porta} @ {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(porta, BAUD_RATE, timeout=3)
        time.sleep(2)   # aguarda ESP32 reiniciar
        print(f"[Serial] Conectado! Aguardando dados do ESP32...")

        while True:
            linha = ser.readline().decode("utf-8", errors="ignore").strip()
            if not linha:
                continue
            try:
                dados = json.loads(linha)
                processar_leitura(dados)
            except json.JSONDecodeError:
                if linha:
                    print(f"[Serial raw] {linha}")

    except serial.SerialException as e:
        print(f"[ERRO] Não foi possível abrir {porta}: {e}")
        print("[FALLBACK] Iniciando modo simulação...")
        simular_esp32()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClimaSat Serial Reader")
    parser.add_argument("--porta",   default=None, help="Porta serial (ex: COM3 ou /dev/ttyUSB0)")
    parser.add_argument("--simular", action="store_true", help="Modo simulação sem ESP32")
    args = parser.parse_args()

    print("=" * 50)
    print("  ClimaSat — Step 5: Leitor ESP32")
    print("=" * 50)

    # Inicia servidor HTTP em thread separada
    t = threading.Thread(target=iniciar_servidor, daemon=True)
    t.start()

    if args.simular:
        simular_esp32()
    else:
        porta = args.porta or detectar_porta()
        if porta:
            ler_serial(porta)
        else:
            print("[AVISO] Nenhuma porta encontrada. Iniciando simulação.")
            simular_esp32()
