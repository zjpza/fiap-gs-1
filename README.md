<p align="center">
  <img src="assets/logo-fiap.png" width="120" alt="FIAP"/>
</p>

<h1 align="center">climAI Farm 🌱🛰️</h1>
<h3 align="center">Agricultura Inteligente com Dados Espaciais e IA</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/ESP32-Arduino-00979D?style=flat&logo=arduino&logoColor=white"/>
  <img src="https://img.shields.io/badge/NASA%20POWER-API-0B3D91?style=flat"/>
  <img src="https://img.shields.io/badge/FIAP-GS%202026.1-ED145B?style=flat"/>
</p>

---

## 🛰️ Sobre o Projeto

O **climAI Farm** é uma POC (Prova de Conceito) desenvolvida para a **Global Solution 2026.1 da FIAP**, tema **Economia Espacial**. O projeto integra dados climáticos de satélites NASA com sensores IoT e Inteligência Artificial para apoiar decisões agrícolas em tempo real.

> **Pergunta respondida:** *Como a tecnologia espacial pode ser utilizada para melhorar a vida das pessoas e tornar processos mais eficientes?*

A plataforma fornece previsões de temperatura e probabilidade de chuva para o dia seguinte, com comparação automática contra a normal climatológica histórica — acessível via dashboard web para qualquer cidade do mundo.

---

## 👨‍🎓 Integrantes

| Nome | RM | GitHub |
|---|---|---|
| Henrique Sanches Silva | RM 570527 | [@HenriqueSanchesSilva](https://github.com/HenriqueSanchesSilva) |
| João Pedro Zavanela Andreu | RM 570231 | [@zjpza](https://github.com/zjpza) |
| Kayck Gabriel Evangelista da Silva | RM 572331 | [@Kayckxz](https://github.com/Kayckxz) |
| Luis Henrique Laurentino Boschi | RM 571352 | [@lhboschi](https://github.com/lhboschi) |
| Patrick Borges de Melo | RM 574030 | [@Trickmelo](https://github.com/Trickmelo) |

**Tutor(a):** Sabrina Otoni | **Coordenador(a):** André Godoi

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                     FONTES DE DADOS                         │
│  NASA POWER API  │  OpenWeatherMap API  │  ESP32 + Sensores  │
│  (histórico 5a)  │  (tempo real, 5min)  │  DHT22 + BMP280    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   PROCESSAMENTO (Python)                    │
│         Pandas • NumPy • Feature Engineering                │
│         Lags • Sazonalidade cíclica • Médias móveis         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   MACHINE LEARNING                          │
│   Ridge Regression (temp T+1)  │  RF Classifier (chuva)    │
│   MAE=1.44°C • R²=0.710        │  AUC-ROC=0.611            │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│               DASHBOARD (Streamlit + Plotly)                │
│  Visão Geral  │  Análise EDA  │  Previsão ML  │  Dados      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.10+
- pip

### Instalação

```bash
git clone https://github.com/zjpza/fiap-gs-1.git
cd fiap-gs-1
pip install -r requirements.txt
```

### Dashboard (interface principal)

```bash
streamlit run src/climasat_dashboard.py
```

Acesse `http://localhost:8501` no navegador. Insira sua chave [OpenWeatherMap](https://openweathermap.org/api) (gratuita) na sidebar para dados em tempo real.

### Pipeline completo (ordem recomendada)

```bash
# 1. Coleta de dados + EDA
python src/climasat_eda.py

# 2. Treinamento dos modelos ML
python src/climasat_ml.py

# 3. (Opcional) Leitor ESP32 em terminal separado
python src/climasat_serial_reader.py --simular   # sem hardware
python src/climasat_serial_reader.py --porta COM3 # com ESP32

# 4. Dashboard
streamlit run src/climasat_dashboard.py
```

### ESP32 (hardware físico)

Abrir `src/esp32_climasat.ino` na Arduino IDE e instalar as bibliotecas:
- DHT sensor library (Adafruit)
- Adafruit BMP280
- ArduinoJson

**Pinagem:**
| Sensor | Pino ESP32 |
|---|---|
| DHT22 DATA | GPIO 4 |
| BMP280 SDA | GPIO 21 |
| BMP280 SCL | GPIO 22 |
| VCC (ambos) | 3.3V |

---

## 📊 Resultados

| Modelo | Tarefa | MAE | RMSE | R² | AUC-ROC |
|---|---|---|---|---|---|
| Ridge Regression | Temperatura T+1 | 1.44°C | 1.80°C | 0.710 | — |
| Random Forest Reg. | Temperatura T+1 | 1.49°C | 1.85°C | 0.693 | — |
| RF Classifier | Chuva T+1 | — | — | — | 0.611 |

---

## 📁 Estrutura de Pastas

```
fiap-gs-1/
├── src/
│   ├── climasat_eda.py              # Coleta NASA POWER + EDA + visualizações
│   ├── climasat_ml.py               # Treinamento Ridge + Random Forest
│   ├── climasat_dashboard.py        # Dashboard Streamlit + Plotly
│   ├── climasat_serial_reader.py    # Leitor Serial ESP32 + servidor HTTP
│   ├── esp32_climasat.ino           # Firmware Arduino ESP32
│   ├── modelo_ridge.pkl             # Modelo serializado
│   ├── modelo_rf_regressor.pkl      # Modelo serializado
│   └── modelo_rf_classificador.pkl  # Modelo serializado
├── data/
│   ├── climasat_dados_limpos.csv    # 1.826 registros NASA POWER
│   └── esp32_leituras.csv           # Leituras do sensor (gerado em runtime)
├── assets/
│   ├── logo-fiap.png
│   ├── 00_painel_resumo.png
│   ├── 01_serie_temperatura.png
│   ├── 03_heatmap_temperatura.png
│   ├── 05_correlacao.png
│   ├── 06_previsto_vs_real.png
│   └── 11_painel_ml.png
├── docs/
│   └── climAI_Farm_GS2026.pdf       # PDF de entrega
├── requirements.txt
└── README.md
```

---

## 🔗 Links

- 🎬 **Vídeo demonstrativo:** `<link_youtube_aqui>` (YouTube — não listado)
- 📄 **PDF de entrega:** `docs/climAI_Farm_GS2026.pdf`
- 🛰️ **NASA POWER API:** https://power.larc.nasa.gov/
- 🌤️ **OpenWeatherMap:** https://openweathermap.org/api

---

## 📋 Licença

[climAI Farm — FIAP Global Solution 2026.1](https://github.com/zjpza/fiap-gs-1) por FIAP está licenciado sob [CC Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/).
