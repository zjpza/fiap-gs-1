# src

Código-fonte do projeto **climAI Farm**.

| Arquivo | Descrição |
|---|---|
| `climasat_eda.py` | Coleta dados da NASA POWER API (com fallback sintético), limpa, enriquece e gera as visualizações da EDA |
| `climasat_ml.py` | Treina os modelos preditivos: Ridge e Random Forest (temperatura T+1) e RF Classifier (chuva T+1); exporta modelos `.pkl` e métricas |
| `climasat_dashboard.py` | Dashboard Streamlit + Plotly — dados em tempo real (OpenWeatherMap), histórico NASA por cidade, previsão ML e dados brutos |
| `climasat_serial_reader.py` | Lê o ESP32 via Serial/USB, salva em CSV e serve um endpoint HTTP para o dashboard; possui modo `--simular` sem hardware |
| `esp32_climasat.ino` | Firmware Arduino para o ESP32 (sensores DHT22 + BMP280) que envia leituras via Serial e Wi-Fi |

## Ordem de execução

```bash
python climasat_eda.py        # 1. Coleta + EDA  → climasat_output/
python climasat_ml.py         # 2. Treina modelos
streamlit run climasat_dashboard.py   # 3. Dashboard
python climasat_serial_reader.py --simular   # 4. (Opcional) ESP32 simulado
```

> **Windows:** os scripts já forçam saída UTF-8 internamente. As dependências estão em `../requirements.txt` (Python 3.10+).
