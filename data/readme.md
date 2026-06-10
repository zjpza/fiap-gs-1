# data

Dados climáticos do projeto **climAI Farm**.

| Arquivo | Descrição |
|---|---|
| `climasat_dados_limpos.csv` | Dataset limpo e enriquecido — 1.826 registros diários (2019–2023) da NASA POWER API para Santo André/SP, com colunas derivadas (amplitude térmica, flag de dia chuvoso, média móvel 30d, features de calendário) |
| `metricas_ml.csv` | Métricas dos modelos treinados (MAE, RMSE, R², Acurácia, AUC-ROC) |

**Origem:** [NASA POWER API](https://power.larc.nasa.gov/) (T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M, WS2M). Em caso de indisponibilidade da API, `climasat_eda.py` gera dados sintéticos baseados nas normais climatológicas do INMET como fallback.

> **Atenção:** não subir dados grandes ou sensíveis. O `.gitignore` ignora `*.csv` por padrão — estes foram versionados intencionalmente por serem leves e necessários à reprodução.
