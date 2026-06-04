"""
ClimaSat — Step 4 (v2): Dashboard com dados em tempo real
=========================================================
Novidades:
  - Seletor de cidade livre (qualquer cidade do mundo)
  - OpenWeatherMap API: temperatura, umidade, pressão e chuva AGORA
  - Comparação automática: leitura atual vs. normal histórica do mês
  - Alerta de anomalia climática
  - NASA POWER atualiza conforme a cidade escolhida
  - ESP32 simulado (substituído por leitura real no Step 5)

Execute:
  streamlit run climasat_dashboard.py

Chave OpenWeatherMap (gratuita):
  https://openweathermap.org/api  →  "Current Weather Data" (plano Free)
  Cole a chave na sidebar ou defina a variável de ambiente OWM_KEY.
"""

import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ClimaSat Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container{padding-top:1.2rem;padding-bottom:.5rem}
.stMetric label{font-size:.75rem!important;color:#888}
.stMetric [data-testid="stMetricValue"]{font-size:1.6rem!important}
.alert-box{padding:10px 14px;border-radius:8px;font-size:.85rem;margin-bottom:8px}
.alert-warn{background:rgba(239,159,39,.12);border-left:3px solid #ef9f27;color:#ef9f27}
.alert-ok  {background:rgba(29,158,117,.12);border-left:3px solid #1d9e75;color:#1d9e75}
.alert-err {background:rgba(232,89,60,.12); border-left:3px solid #e8593c;color:#e8593c}
.now-card  {background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;margin-bottom:12px}
</style>
""", unsafe_allow_html=True)

# Paleta
CR = "#e8593c"; CB = "#3b8bd4"; CG = "#1d9e75"; CP = "#7f77dd"; CY = "#ef9f27"
BG = "#0d1117"; SF = "#161b22"

# Cidades populares com coordenadas
CIDADES = {
    "Santo André, SP":      (-23.6639, -46.5432),
    "São Paulo, SP":        (-23.5505, -46.6333),
    "Rio de Janeiro, RJ":   (-22.9068, -43.1729),
    "Belo Horizonte, MG":   (-19.9167, -43.9345),
    "Curitiba, PR":         (-25.4284, -49.2733),
    "Porto Alegre, RS":     (-30.0277, -51.2287),
    "Salvador, BA":         (-12.9714, -38.5014),
    "Fortaleza, CE":        (-3.7172,  -38.5433),
    "Manaus, AM":           (-3.1190,  -60.0217),
    "Brasília, DF":         (-15.7801, -47.9292),
    "Outra cidade...":      None,
}

MESES = ["Jan","Fev","Mar","Abr","Mai","Jun",
         "Jul","Ago","Set","Out","Nov","Dez"]

# ──────────────────────────────────────────────
# CARREGAMENTO DE DADOS
# ──────────────────────────────────────────────
@st.cache_data
def carregar_dataset():
    path = "climasat_output/climasat_dados_limpos.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="data", parse_dates=True)
    df.sort_index(inplace=True)
    return df

@st.cache_resource
def carregar_modelos():
    modelos = {}
    for nome, arq in [("reg","modelo_rf_regressor.pkl"),("clf","modelo_rf_classificador.pkl")]:
        p = f"climasat_output/{arq}"
        if os.path.exists(p):
            modelos[nome] = joblib.load(p)
    return modelos

df_hist  = carregar_dataset()
modelos  = carregar_modelos()

# ──────────────────────────────────────────────
# OPENWEATHERMAP — DADOS EM TEMPO REAL
# ──────────────────────────────────────────────
@st.cache_data(ttl=300)   # atualiza a cada 5 minutos
def buscar_clima_atual(lat, lon, api_key):
    """Busca temperatura, umidade, pressão e condição atual via OWM."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=pt_br"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()
        return {
            "ok":          True,
            "temp":        round(d["main"]["temp"],     1),
            "temp_max":    round(d["main"]["temp_max"], 1),
            "temp_min":    round(d["main"]["temp_min"], 1),
            "umidade":     d["main"]["humidity"],
            "pressao":     d["main"]["pressure"],
            "vento":       round(d["wind"]["speed"],    1),
            "descricao":   d["weather"][0]["description"].capitalize(),
            "icone":       d["weather"][0]["icon"],
            "cidade":      d.get("name",""),
            "chuva_1h":    d.get("rain",{}).get("1h", 0.0),
            "timestamp":   datetime.utcfromtimestamp(d["dt"]).strftime("%H:%M UTC"),
        }
    except Exception as e:
        return {"ok": False, "erro": str(e)}


@st.cache_data(ttl=3600)
def buscar_geocode(cidade_nome, api_key):
    """Geocodifica nome de cidade usando OWM Geocoding API."""
    url = (
        f"https://api.openweathermap.org/geo/1.0/direct"
        f"?q={requests.utils.quote(cidade_nome)}&limit=1&appid={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        res = r.json()
        if res:
            return res[0]["lat"], res[0]["lon"], res[0].get("local_names",{}).get("pt", res[0]["name"])
        return None, None, None
    except:
        return None, None, None


@st.cache_data(ttl=86400)
def buscar_historico_nasa(lat, lon):
    """Busca dados históricos NASA POWER para a localidade."""
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS2M"
        f"&community=RE&longitude={lon}&latitude={lat}"
        f"&start=20190101&end=20231231&format=JSON"
    )
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        series = r.json()["properties"]["parameter"]
        df = pd.DataFrame(series)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.replace(-999.0, np.nan, inplace=True)
        df.interpolate(method="time", limit=3, inplace=True)
        df["mes"] = df.index.month
        df["ano"] = df.index.year
        return df, True
    except:
        return df_hist, False   # fallback p/ Santo André


# ──────────────────────────────────────────────
# NORMAL CLIMATOLÓGICA (média por mês)
# ──────────────────────────────────────────────
def normal_climatologica(df):
    return df.groupby("mes")[["T2M","T2M_MAX","T2M_MIN","PRECTOTCORR","RH2M"]].mean()

def anomalia(valor_atual, normal_mes, col):
    diff  = valor_atual - normal_mes[col]
    pct   = diff / normal_mes[col] * 100 if normal_mes[col] else 0
    return diff, pct


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ ClimaSat")
    st.caption("Análise climática com dados de satélite")
    st.divider()

    # Seletor de cidade
    st.subheader("📍 Localidade")
    cidade_sel = st.selectbox("Cidade", list(CIDADES.keys()), index=0)

    lat, lon = None, None

    if cidade_sel == "Outra cidade...":
        cidade_texto = st.text_input(
            "Digite o nome da cidade",
            placeholder="ex: Campinas, SP  /  Buenos Aires  /  Lisbon"
        )
    else:
        cidade_texto = None
        lat, lon = CIDADES[cidade_sel]

    st.divider()

    # Chave API
    st.subheader("🔑 OpenWeatherMap API")
    api_key = st.text_input(
        "Chave API (gratuita)",
        value=os.getenv("OWM_KEY",""),
        type="password",
        help="Crie em openweathermap.org/api — plano Free é suficiente",
    )

    st.caption("[Criar chave grátis ↗](https://openweathermap.org/api)")
    st.divider()

    # Navegação
    st.subheader("Navegação")
    pagina = st.radio("", [
        "🏠 Visão Geral",
        "📊 Análise EDA",
        "🤖 Previsão ML",
        "🗃️ Dados Brutos",
    ], label_visibility="collapsed")

    # ESP32 status
    st.divider()
    st.caption("📡 ESP32 / Sensor local")
    esp_on = st.toggle("Simular leitura ESP32", value=True)
    if esp_on:
        st.caption("🟢 Sensor ativo (Step 5)")
    else:
        st.caption("🔴 Sem sensor conectado")


# ──────────────────────────────────────────────
# RESOLUÇÃO DE LOCALIDADE
# ──────────────────────────────────────────────
cidade_nome_display = cidade_sel

if cidade_texto and api_key:
    with st.spinner(f"Buscando '{cidade_texto}'..."):
        lat_geo, lon_geo, nome_geo = buscar_geocode(cidade_texto, api_key)
    if lat_geo:
        lat, lon = lat_geo, lon_geo
        cidade_nome_display = nome_geo or cidade_texto
    else:
        st.warning("Cidade não encontrada. Usando Santo André como fallback.")
        lat, lon = CIDADES["Santo André, SP"]
elif cidade_texto and not api_key:
    st.info("Insira a chave API para buscar cidades personalizadas.")
    lat, lon = CIDADES["Santo André, SP"]

if lat is None:
    lat, lon = CIDADES["Santo André, SP"]

# ──────────────────────────────────────────────
# DADOS DA LOCALIDADE
# ──────────────────────────────────────────────
with st.spinner("Carregando dados históricos..."):
    df_local, nasa_ok = buscar_historico_nasa(lat, lon)

normal = normal_climatologica(df_local)
mes_atual = datetime.now().month

clima_atual = None
if api_key:
    with st.spinner("Buscando clima atual..."):
        clima_atual = buscar_clima_atual(lat, lon, api_key)

# ──────────────────────────────────────────────
# PÁGINA 1 — VISÃO GERAL
# ──────────────────────────────────────────────
if pagina == "🏠 Visão Geral":
    st.title(f"🛰️ ClimaSat — {cidade_nome_display}")

    fonte_badge = "✅ NASA POWER" if nasa_ok else "⚠️ Dados de referência (Santo André)"
    st.caption(f"{fonte_badge}  ·  Histórico 2019–2023  ·  {len(df_local):,} registros")

    # ── Bloco tempo real ──────────────────────
    if clima_atual and clima_atual["ok"]:
        now = clima_atual
        st.subheader(f"🌤️ Agora em {now['cidade']} — {now['timestamp']}")

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("🌡️ Temperatura",  f"{now['temp']} °C",
                  delta=f"{now['temp'] - normal.loc[mes_atual,'T2M']:+.1f}°C vs média hist.")
        c2.metric("💧 Umidade",      f"{now['umidade']} %",
                  delta=f"{now['umidade'] - normal.loc[mes_atual,'RH2M']:+.0f}% vs média hist.")
        c3.metric("🔵 Pressão",      f"{now['pressao']} hPa")
        c4.metric("💨 Vento",        f"{now['vento']} m/s")
        c5.metric("🌧️ Chuva 1h",     f"{now['chuva_1h']:.1f} mm")

        # Alertas de anomalia
        diff_t, pct_t = anomalia(now["temp"], normal.loc[mes_atual], "T2M")
        diff_u, pct_u = anomalia(now["umidade"], normal.loc[mes_atual], "RH2M")

        if abs(diff_t) >= 4:
            tipo = "acima" if diff_t > 0 else "abaixo"
            st.markdown(
                f'<div class="alert-box alert-err">⚠️ Anomalia térmica: temperatura {abs(diff_t):.1f}°C '
                f'{tipo} da normal histórica de {MESES[mes_atual-1]} '
                f'({normal.loc[mes_atual,"T2M"]:.1f}°C)</div>',
                unsafe_allow_html=True
            )
        elif abs(diff_t) >= 2:
            tipo = "acima" if diff_t > 0 else "abaixo"
            st.markdown(
                f'<div class="alert-box alert-warn">🌡️ Temperatura {abs(diff_t):.1f}°C {tipo} '
                f'da normal histórica de {MESES[mes_atual-1]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="alert-box alert-ok">✅ Temperatura dentro da faixa normal para '
                f'{MESES[mes_atual-1]} ({normal.loc[mes_atual,"T2M"]:.1f}°C histórico)</div>',
                unsafe_allow_html=True
            )

        # ESP32 simulado
        if esp_on:
            st.subheader("📡 Leitura ESP32 (sensor local)")
            e1,e2,e3 = st.columns(3)
            # Simula leitura com pequena variação em relação ao OWM
            t_esp  = round(now["temp"]    + np.random.normal(0,.3),  1)
            u_esp  = round(now["umidade"] + np.random.normal(0, 1),  0)
            p_esp  = round(now["pressao"] + np.random.normal(0,.5),  1)
            e1.metric("🌡️ DHT22 — Temp.",  f"{t_esp} °C",   delta=f"{t_esp-now['temp']:+.1f}°C vs OWM")
            e2.metric("💧 DHT22 — Umid.",  f"{u_esp:.0f} %", delta=f"{u_esp-now['umidade']:+.0f}% vs OWM")
            e3.metric("🔵 BMP280 — Press.", f"{p_esp} hPa",  delta=f"{p_esp-now['pressao']:+.1f} vs OWM")
            st.caption("⚙️ Dados simulados — sensor físico integrado no Step 5")

        st.divider()

    elif api_key and clima_atual and not clima_atual["ok"]:
        st.error(f"❌ Erro ao buscar clima atual: {clima_atual['erro']}")
    elif not api_key:
        st.info("💡 Insira sua chave OpenWeatherMap na sidebar para ver dados em tempo real.")

    # ── KPIs históricos ───────────────────────
    st.subheader("📊 Estatísticas históricas (2019–2023)")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🌡️ Temp. média",    f"{df_local['T2M'].mean():.1f} °C")
    c2.metric("🔥 Máx. abs.",      f"{df_local['T2M_MAX'].max():.1f} °C")
    c3.metric("❄️ Mín. abs.",      f"{df_local['T2M_MIN'].min():.1f} °C")
    c4.metric("🌧️ Precip. total",  f"{df_local['PRECTOTCORR'].sum():.0f} mm")
    c5.metric("💧 Umid. média",    f"{df_local['RH2M'].mean():.1f} %")

    # ── Série histórica ───────────────────────
    st.subheader("Série histórica de temperatura")
    mm30 = df_local["T2M"].rolling(30, center=True).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_local.index, y=df_local["T2M_MAX"], name="Máxima",
                             line=dict(color=CR, width=0), fill=None, mode="lines", opacity=.3))
    fig.add_trace(go.Scatter(x=df_local.index, y=df_local["T2M_MIN"], name="Amplitude",
                             fill="tonexty", fillcolor="rgba(232,89,60,.1)",
                             line=dict(color=CR, width=0), mode="lines"))
    fig.add_trace(go.Scatter(x=df_local.index, y=df_local["T2M"], name="Média diária",
                             line=dict(color=CR, width=.8), opacity=.5, mode="lines"))
    fig.add_trace(go.Scatter(x=df_local.index, y=mm30, name="Média móvel 30d",
                             line=dict(color="#A32D2D", width=2.5)))
    if clima_atual and clima_atual["ok"]:
        fig.add_hline(y=clima_atual["temp"], line_dash="dash",
                      line_color=CY, annotation_text=f"Agora: {clima_atual['temp']}°C")
    fig.update_layout(height=340, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                      font_color="#e6edf3", margin=dict(l=0,r=0,t=10,b=0),
                      legend=dict(orientation="h", y=1.08), hovermode="x unified",
                      yaxis_title="Temperatura (°C)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Normal climatológica do mês atual ─────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Normal climatológica — {MESES[mes_atual-1]}")
        fig2 = go.Figure()
        vals = [normal.loc[m,"T2M"] for m in range(1,13)]
        cores = [CR if m==mes_atual else CB for m in range(1,13)]
        fig2.add_trace(go.Bar(x=MESES, y=vals, marker_color=cores, name="Temp. média"))
        if clima_atual and clima_atual["ok"]:
            fig2.add_hline(y=clima_atual["temp"], line_dash="dash", line_color=CY,
                           annotation_text=f"Agora: {clima_atual['temp']}°C")
        fig2.update_layout(height=260, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                           font_color="#e6edf3", margin=dict(l=0,r=0,t=10,b=0),
                           yaxis_title="°C")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Precipitação mensal acumulada")
        chuva_m = df_local.groupby(pd.Grouper(freq="ME"))["PRECTOTCORR"].sum().reset_index()
        chuva_m.columns = ["data", "PRECTOTCORR"]
        fig3 = px.bar(chuva_m, x="data", y="PRECTOTCORR",
                      color_discrete_sequence=[CB])
        fig3.update_layout(height=260, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                           font_color="#e6edf3", margin=dict(l=0,r=0,t=10,b=0),
                           yaxis_title="mm")
        st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────
# PÁGINA 2 — EDA
# ──────────────────────────────────────────────
elif pagina == "📊 Análise EDA":
    st.title(f"📊 Análise EDA — {cidade_nome_display}")

    tab1, tab2, tab3 = st.tabs(["🌡️ Temperatura", "🌧️ Precipitação", "🔗 Correlações"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Heatmap temperatura média")
            pivot = df_local.pivot_table(values="T2M", index="mes", columns="ano", aggfunc="mean")
            fig = px.imshow(pivot, color_continuous_scale="RdYlBu_r",
                            labels=dict(x="Ano",y="Mês",color="°C"),
                            y=MESES, aspect="auto", text_auto=".1f")
            fig.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Boxplot mensal")
            df_box = df_local.copy()
            df_box["mes_nome"] = df_box["mes"].apply(lambda x: MESES[x-1])
            fig = px.box(df_box, x="mes_nome", y="T2M",
                         category_orders={"mes_nome":MESES},
                         color="mes_nome",
                         color_discrete_sequence=px.colors.sequential.RdBu_r,
                         labels={"T2M":"°C","mes_nome":"Mês"})
            fig.update_layout(height=360, showlegend=False,
                              margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        prec_mes = df_local.groupby("mes")["PRECTOTCORR"].agg(["mean","sum"]).reset_index()
        prec_mes["mes_nome"] = prec_mes["mes"].apply(lambda x: MESES[x-1])
        fig = px.bar(prec_mes, x="mes_nome", y="mean",
                     labels={"mean":"Precip. média diária (mm)","mes_nome":"Mês"},
                     color_discrete_sequence=[CB])
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        cols = ["T2M","T2M_MAX","T2M_MIN","PRECTOTCORR","RH2M","WS2M"]
        labels = ["Temp.média","Temp.máx.","Temp.mín.","Precipit.","Umidade","Vento"]
        corr = df_local[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig = px.imshow(corr.where(~mask), x=labels, y=labels,
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        text_auto=".2f", aspect="auto")
        fig.update_layout(height=460, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# PÁGINA 3 — PREVISÃO ML
# ──────────────────────────────────────────────
elif pagina == "🤖 Previsão ML":
    st.title("🤖 Previsão ML")

    # Pré-preenche com dados reais se disponíveis
    t_default = clima_atual["temp"]    if (clima_atual and clima_atual["ok"]) else 22.0
    u_default = clima_atual["umidade"] if (clima_atual and clima_atual["ok"]) else 78.0
    p_default = clima_atual["chuva_1h"]if (clima_atual and clima_atual["ok"]) else 0.0

    if clima_atual and clima_atual["ok"]:
        st.success(f"✅ Sliders pré-preenchidos com leitura atual de {cidade_nome_display} ({clima_atual['timestamp']})")

    tab1, tab2 = st.tabs(["🌡️ Prever Temperatura", "🌧️ Prever Chuva"])

    FEAT_REG = ["T2M","T2M_MAX","T2M_MIN","amplitude","RH2M","WS2M","PRECTOTCORR",
                "mes_sin","mes_cos","dia_sem_sin","dia_sem_cos",
                "T2M_lag1","T2M_lag2","T2M_lag3","T2M_lag7","T2M_mm7","T2M_mm14"]
    FEAT_CLF = ["T2M","T2M_MAX","T2M_MIN","RH2M","WS2M","PRECTOTCORR",
                "mes_sin","mes_cos",
                "PRECTOTCORR_lag1","PRECTOTCORR_lag2","PRECTOTCORR_lag3",
                "PREC_mm7","T2M_mm7"]

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            t_med = st.slider("Temp. média hoje (°C)",  10.0, 40.0, float(t_default), .1)
            t_max = st.slider("Temp. máxima hoje (°C)", 15.0, 45.0, float(t_default)+5, .1)
            t_min = st.slider("Temp. mínima hoje (°C)",  0.0, 35.0, float(t_default)-5, .1)
        with c2:
            umid   = st.slider("Umidade (%)",              20.0, 100.0, float(u_default), 1.0)
            vento  = st.slider("Vento (m/s)",               0.0,  20.0, 3.5, .1)
            precip = st.slider("Precipitação hoje (mm)",    0.0,  80.0, float(p_default), .5)
        with c3:
            mes   = st.selectbox("Mês", range(1,13), format_func=lambda x: MESES[x-1],
                                  index=mes_atual-1)
            dia_s = st.selectbox("Dia da semana",
                                  ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"],
                                  index=datetime.now().weekday())

        if "reg" in modelos:
            dia_idx = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"].index(dia_s)
            inp = {"T2M":t_med,"T2M_MAX":t_max,"T2M_MIN":t_min,"amplitude":t_max-t_min,
                   "RH2M":umid,"WS2M":vento,"PRECTOTCORR":precip,
                   "mes_sin":np.sin(2*np.pi*mes/12),"mes_cos":np.cos(2*np.pi*mes/12),
                   "dia_sem_sin":np.sin(2*np.pi*dia_idx/7),"dia_sem_cos":np.cos(2*np.pi*dia_idx/7),
                   "T2M_lag1":t_med,"T2M_lag2":t_med-.3,"T2M_lag3":t_med-.6,"T2M_lag7":t_med-.8,
                   "T2M_mm7":t_med-.5,"T2M_mm14":t_med-.4}
            pred = modelos["reg"].predict(pd.DataFrame([inp])[FEAT_REG])[0]
            st.divider()
            r1,r2,r3 = st.columns(3)
            r1.metric("🌡️ Amanhã", f"{pred:.1f} °C", delta=f"{pred-t_med:+.1f}°C vs hoje")
            norm_t = normal.loc[mes,"T2M"]
            r2.metric("📊 Normal histórica", f"{norm_t:.1f} °C",
                      delta=f"{pred-norm_t:+.1f}°C vs normal")
            r3.metric("📍 Localidade", cidade_nome_display)
        else:
            st.warning("Modelos não encontrados. Execute o Step 3 primeiro.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            t_c  = st.slider("Temp. hoje (°C)",     10.0, 40.0, float(t_default), .1, key="tc")
            u_c  = st.slider("Umidade (%)",          20.0,100.0, float(u_default), 1.0, key="uc")
            p_c  = st.slider("Precip. hoje (mm)",     0.0, 80.0, float(p_default),  .5, key="pc")
        with c2:
            p1_c = st.slider("Precip. ontem (mm)",    0.0, 80.0, 0.0, .5)
            p2_c = st.slider("Precip. anteontem (mm)",0.0, 80.0, 0.0, .5)
            mes_c= st.selectbox("Mês", range(1,13), format_func=lambda x: MESES[x-1],
                                 index=mes_atual-1, key="mc")
        if "clf" in modelos:
            inp_c = {"T2M":t_c,"T2M_MAX":t_c+5,"T2M_MIN":t_c-5,"RH2M":u_c,"WS2M":3.5,
                     "PRECTOTCORR":p_c,"mes_sin":np.sin(2*np.pi*mes_c/12),
                     "mes_cos":np.cos(2*np.pi*mes_c/12),
                     "PRECTOTCORR_lag1":p1_c,"PRECTOTCORR_lag2":p2_c,"PRECTOTCORR_lag3":0.0,
                     "PREC_mm7":p_c+p1_c+p2_c,"T2M_mm7":t_c-.3}
            proba = modelos["clf"].predict_proba(pd.DataFrame([inp_c])[FEAT_CLF])[0][1]
            st.divider()
            col_a, col_b = st.columns(2)
            col_a.metric("Probabilidade de chuva amanhã", f"{proba*100:.1f}%")
            col_a.metric("Previsão", "🌧️ Dia chuvoso" if proba>=.5 else "☀️ Dia seco")

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba*100,
                number={"suffix":"%"},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":CB if proba>=.5 else CY},
                       "threshold":{"line":{"color":"red","width":2},
                                    "thickness":.75,"value":50},
                       "steps":[{"range":[0,30],"color":"#1a2a1a"},
                                 {"range":[30,60],"color":"#2a2a1a"},
                                 {"range":[60,100],"color":"#1a1a2a"}]}
            ))
            fig_g.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)",
                                 font_color="#e6edf3", margin=dict(l=20,r=20,t=20,b=0))
            col_b.plotly_chart(fig_g, use_container_width=True)
        else:
            st.warning("Modelos não encontrados. Execute o Step 3 primeiro.")


# ──────────────────────────────────────────────
# PÁGINA 4 — DADOS BRUTOS
# ──────────────────────────────────────────────
elif pagina == "🗃️ Dados Brutos":
    st.title(f"🗃️ Dados Brutos — {cidade_nome_display}")

    if clima_atual and clima_atual["ok"]:
        st.subheader("🔴 Leitura atual (tempo real)")
        now = clima_atual
        dados_rt = pd.DataFrame([{
            "Fonte": "OpenWeatherMap",
            "Horário": now["timestamp"],
            "Temp. (°C)": now["temp"],
            "Umidade (%)": now["umidade"],
            "Pressão (hPa)": now["pressao"],
            "Vento (m/s)": now["vento"],
            "Condição": now["descricao"],
        }])
        st.dataframe(dados_rt, use_container_width=True, hide_index=True)

    st.subheader("📅 Histórico NASA POWER (últimos 30 dias disponíveis)")
    cols_show = ["T2M","T2M_MAX","T2M_MIN","PRECTOTCORR","RH2M","WS2M"]
    df_show = df_local[cols_show].tail(30).copy()
    df_show.columns = ["T.Média","T.Máx.","T.Mín.","Precip.","Umidade","Vento"]
    st.dataframe(
        df_show.style
               .background_gradient(subset=["T.Média"], cmap="RdYlBu_r")
               .format("{:.2f}"),
        use_container_width=True, height=380
    )

    st.download_button(
        "⬇️ Baixar CSV completo",
        data=df_local[cols_show].to_csv().encode("utf-8"),
        file_name=f"climasat_{cidade_nome_display.replace(' ','_').replace(',','')}.csv",
        mime="text/csv",
    )

    if os.path.exists("climasat_output/metricas_ml.csv"):
        st.subheader("📈 Métricas ML (Step 3)")
        st.dataframe(pd.read_csv("climasat_output/metricas_ml.csv"),
                     use_container_width=True, hide_index=True)
