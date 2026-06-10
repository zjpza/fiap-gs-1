"""
ClimaSat — Step 3: Machine Learning — Modelos Preditivos e Métricas
====================================================================
Modelos treinados:
  1. Ridge Regression  → previsão de temperatura (T+1 dia)
  2. Random Forest     → previsão de temperatura (T+1 dia)  [comparativo]
  3. Random Forest     → classificação de dia chuvoso (T+1 dia)

Métricas avaliadas:
  Regressão   : MAE, RMSE, R²
  Classificação: Accuracy, Precision, Recall, F1, AUC-ROC

Entrada : climasat_output/climasat_dados_limpos.csv  (gerado no Step 2)
Saídas  : climasat_output/  (gráficos + modelos .pkl)
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Garante saída UTF-8 no terminal (Windows usa cp1252 por padrão)
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.linear_model       import Ridge
from sklearn.ensemble           import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing      import StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# ──────────────────────────────────────────────
# 0. CONFIGURAÇÕES
# ──────────────────────────────────────────────
INPUT_CSV  = "climasat_output/climasat_dados_limpos.csv"
OUTPUT_DIR = "climasat_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42

PALETTE = {
    "real":     "#E8593C",
    "pred_lr":  "#3B8BD4",
    "pred_rf":  "#1D9E75",
    "chuva":    "#7F77DD",
    "fundo":    "#F8F7F2",
    "texto":    "#2C2C2A",
    "grid":     "#D3D1C7",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": PALETTE["fundo"],
    "axes.facecolor":   PALETTE["fundo"],
    "axes.labelcolor":  PALETTE["texto"],
    "xtick.color":      PALETTE["texto"],
    "ytick.color":      PALETTE["texto"],
    "text.color":       PALETTE["texto"],
})


# ──────────────────────────────────────────────
# 1. CARREGAMENTO E FEATURE ENGINEERING
# ──────────────────────────────────────────────
def carregar_e_preparar(path):
    df = pd.read_csv(path, index_col="data", parse_dates=True)
    df.sort_index(inplace=True)

    # Features temporais cíclicas (captura sazonalidade)
    df["mes_sin"]     = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"]     = np.cos(2 * np.pi * df["mes"] / 12)
    df["dia_sem_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_sem_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)

    # Lags (valores dos dias anteriores)
    for lag in [1, 2, 3, 7]:
        df[f"T2M_lag{lag}"]         = df["T2M"].shift(lag)
        df[f"PRECTOTCORR_lag{lag}"] = df["PRECTOTCORR"].shift(lag)

    # Médias/acumulados móveis
    df["T2M_mm7"]  = df["T2M"].rolling(7).mean()
    df["T2M_mm14"] = df["T2M"].rolling(14).mean()
    df["PREC_mm7"] = df["PRECTOTCORR"].rolling(7).sum()

    # Targets: valor do DIA SEGUINTE (T+1)
    df["TARGET_TEMP"]  = df["T2M"].shift(-1)
    df["TARGET_CHUVA"] = df["dia_chuvoso"].shift(-1)

    df.dropna(inplace=True)
    print(f"[Dados] {len(df)} amostras após feature engineering.")
    return df


def separar_features(df):
    feat_reg = [
        "T2M", "T2M_MAX", "T2M_MIN", "amplitude",
        "RH2M", "WS2M", "PRECTOTCORR",
        "mes_sin", "mes_cos", "dia_sem_sin", "dia_sem_cos",
        "T2M_lag1", "T2M_lag2", "T2M_lag3", "T2M_lag7",
        "T2M_mm7", "T2M_mm14",
    ]
    feat_clf = [
        "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M",
        "PRECTOTCORR",
        "mes_sin", "mes_cos",
        "PRECTOTCORR_lag1", "PRECTOTCORR_lag2", "PRECTOTCORR_lag3",
        "PREC_mm7", "T2M_mm7",
    ]
    return feat_reg, feat_clf


# ──────────────────────────────────────────────
# 2. REGRESSÃO — previsão de temperatura
# ──────────────────────────────────────────────
def treinar_regressao(df, features):
    X = df[features]
    y = df["TARGET_TEMP"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=False
    )

    # Ridge Regression
    pipe_lr = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    pipe_lr.fit(X_train, y_train)
    pred_lr = pipe_lr.predict(X_test)

    # Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_leaf=4, random_state=SEED, n_jobs=-1
    )
    rf_reg.fit(X_train, y_train)
    pred_rf = rf_reg.predict(X_test)

    def metricas(y_true, y_pred, nome):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        print(f"\n  [{nome}]  MAE={mae:.3f}°C  RMSE={rmse:.3f}°C  R²={r2:.4f}")
        return {"nome": nome, "MAE": mae, "RMSE": rmse, "R2": r2}

    print("\n── Métricas — Regressão ─────────────────────────")
    m_lr = metricas(y_test, pred_lr, "Ridge Regression")
    m_rf = metricas(y_test, pred_rf, "Random Forest Regressor")

    cv = cross_val_score(rf_reg, X, y, cv=5, scoring="r2")
    print(f"  [RF Cross-val R²] {cv.mean():.4f} ± {cv.std():.4f}")

    importancias = pd.Series(rf_reg.feature_importances_, index=features).sort_values(ascending=False)

    joblib.dump(pipe_lr, f"{OUTPUT_DIR}/modelo_ridge.pkl")
    joblib.dump(rf_reg,  f"{OUTPUT_DIR}/modelo_rf_regressor.pkl")
    print(f"[Modelos] Salvos em {OUTPUT_DIR}/")

    return {
        "X_test": X_test, "y_test": y_test,
        "pred_lr": pred_lr, "pred_rf": pred_rf,
        "metricas": [m_lr, m_rf],
        "importancias": importancias,
        "dates_test": X_test.index,
    }


# ──────────────────────────────────────────────
# 3. CLASSIFICAÇÃO — dia chuvoso?
# ──────────────────────────────────────────────
def treinar_classificacao(df, features):
    X = df[features]
    y = df["TARGET_CHUVA"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=False
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    pred_clf   = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred_clf)
    auc = roc_auc_score(y_test, pred_proba)

    print("\n── Métricas — Classificação ─────────────────────")
    print(f"  Acurácia={acc:.4f}  AUC-ROC={auc:.4f}")
    print(classification_report(y_test, pred_clf,
                                 target_names=["Dia seco","Dia chuvoso"], digits=3))

    importancias_clf = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    joblib.dump(clf, f"{OUTPUT_DIR}/modelo_rf_classificador.pkl")

    return {
        "clf": clf, "X_test": X_test, "y_test": y_test,
        "pred_clf": pred_clf, "pred_proba": pred_proba,
        "acc": acc, "auc": auc, "importancias": importancias_clf,
    }


# ──────────────────────────────────────────────
# 4. VISUALIZAÇÕES
# ──────────────────────────────────────────────
def grafico_previsto_real(res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Temperatura prevista × real (conjunto de teste)", fontsize=14, fontweight="bold")
    for ax, pred, cor, nome in zip(
        axes,
        [res["pred_lr"], res["pred_rf"]],
        [PALETTE["pred_lr"], PALETTE["pred_rf"]],
        ["Ridge Regression", "Random Forest"]
    ):
        ax.scatter(res["y_test"], pred, alpha=0.25, s=8, color=cor, edgecolors="none")
        lims = [res["y_test"].min()-1, res["y_test"].max()+1]
        ax.plot(lims, lims, "--", color=PALETTE["real"], lw=1.5, label="Perfeito")
        mae = mean_absolute_error(res["y_test"], pred)
        r2  = r2_score(res["y_test"], pred)
        ax.set_title(f"{nome}\nMAE={mae:.2f}°C  R²={r2:.4f}")
        ax.set_xlabel("Real (°C)"); ax.set_ylabel("Previsto (°C)")
        ax.set_facecolor(PALETTE["fundo"]); ax.legend(fontsize=9)
    fig.tight_layout()
    path = f"{OUTPUT_DIR}/06_previsto_vs_real.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Gráfico] Salvo: {path}")


def grafico_serie_predicao(res):
    n = min(180, len(res["y_test"]))
    datas = res["dates_test"][-n:]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(datas, res["y_test"].values[-n:],  color=PALETTE["real"],    lw=1.5, label="Real")
    ax.plot(datas, res["pred_lr"][-n:],        color=PALETTE["pred_lr"], lw=1.0, ls="--", alpha=0.8, label="Ridge")
    ax.plot(datas, res["pred_rf"][-n:],        color=PALETTE["pred_rf"], lw=1.0, ls=":",  alpha=0.8, label="Random Forest")
    ax.set_title("Previsão de temperatura — últimos 180 dias do teste", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Temperatura (°C)"); ax.legend(fontsize=9)
    ax.set_facecolor(PALETTE["fundo"]); fig.tight_layout()
    path = f"{OUTPUT_DIR}/07_serie_predicao.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Gráfico] Salvo: {path}")


def grafico_residuos(res):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Distribuição dos resíduos", fontsize=14, fontweight="bold")
    for ax, residuos, cor, nome in zip(
        axes,
        [res["y_test"].values - res["pred_lr"], res["y_test"].values - res["pred_rf"]],
        [PALETTE["pred_lr"], PALETTE["pred_rf"]],
        ["Ridge Regression", "Random Forest"]
    ):
        ax.hist(residuos, bins=50, color=cor, edgecolor="white", lw=0.3, alpha=0.85)
        ax.axvline(0, color=PALETTE["real"], lw=1.5, ls="--")
        ax.set_title(f"{nome}  (std={residuos.std():.2f}°C)")
        ax.set_xlabel("Resíduo (°C)"); ax.set_ylabel("Frequência")
        ax.set_facecolor(PALETTE["fundo"])
    fig.tight_layout()
    path = f"{OUTPUT_DIR}/08_residuos.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Gráfico] Salvo: {path}")


def grafico_importancia(imp, titulo, sufixo):
    top = imp.head(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top.index[::-1], top.values[::-1],
                   color=PALETTE["pred_rf"], edgecolor="white", lw=0.3)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9, color=PALETTE["texto"])
    ax.set_title(titulo, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Importância (Gini)"); ax.set_facecolor(PALETTE["fundo"])
    fig.tight_layout()
    path = f"{OUTPUT_DIR}/{sufixo}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Gráfico] Salvo: {path}")


def grafico_classificacao(res_clf):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Classificador de dia chuvoso — Random Forest", fontsize=14, fontweight="bold")
    cm = confusion_matrix(res_clf["y_test"], res_clf["pred_clf"])
    ConfusionMatrixDisplay(cm, display_labels=["Dia seco","Dia chuvoso"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title(f"Matriz de confusão  (Acc={res_clf['acc']:.3f})")
    axes[0].set_facecolor(PALETTE["fundo"])
    fpr, tpr, _ = roc_curve(res_clf["y_test"], res_clf["pred_proba"])
    axes[1].plot(fpr, tpr, color=PALETTE["chuva"], lw=2, label=f"AUC={res_clf['auc']:.3f}")
    axes[1].plot([0,1],[0,1],"--", color=PALETTE["grid"], lw=1)
    axes[1].fill_between(fpr, tpr, alpha=0.08, color=PALETTE["chuva"])
    axes[1].set_xlabel("Taxa de Falsos Positivos"); axes[1].set_ylabel("Taxa de Verdadeiros Positivos")
    axes[1].set_title("Curva ROC"); axes[1].legend(fontsize=10)
    axes[1].set_facecolor(PALETTE["fundo"]); fig.tight_layout()
    path = f"{OUTPUT_DIR}/10_classificacao_roc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Gráfico] Salvo: {path}")


def grafico_painel_ml(res_reg, res_clf):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("ClimaSat — Painel ML · Modelos Preditivos", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(res_reg["y_test"], res_reg["pred_rf"], alpha=0.2, s=6,
                color=PALETTE["pred_rf"], edgecolors="none")
    lims = [res_reg["y_test"].min()-1, res_reg["y_test"].max()+1]
    ax1.plot(lims, lims, "--", color=PALETTE["real"], lw=1.5)
    r2 = r2_score(res_reg["y_test"], res_reg["pred_rf"])
    ax1.set_title(f"RF Regressor\nR²={r2:.4f}")
    ax1.set_xlabel("Real (°C)"); ax1.set_ylabel("Previsto (°C)")
    ax1.set_facecolor(PALETTE["fundo"])

    ax2 = fig.add_subplot(gs[0, 1:])
    n = 90; datas = res_reg["dates_test"][-n:]
    ax2.plot(datas, res_reg["y_test"].values[-n:], color=PALETTE["real"], lw=1.5, label="Real")
    ax2.plot(datas, res_reg["pred_rf"][-n:], color=PALETTE["pred_rf"], lw=1, ls=":", label="RF Pred.")
    ax2.set_title("Previsão vs. real — 90 dias"); ax2.legend(fontsize=8)
    ax2.set_facecolor(PALETTE["fundo"])

    ax3 = fig.add_subplot(gs[1, 0])
    imp = res_reg["importancias"].head(8)
    ax3.barh(imp.index[::-1], imp.values[::-1], color=PALETTE["pred_rf"], edgecolor="white", lw=0.3)
    ax3.set_title("Features — Temperatura"); ax3.set_facecolor(PALETTE["fundo"])

    ax4 = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(res_clf["y_test"], res_clf["pred_clf"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Seco","Chuva"], yticklabels=["Seco","Chuva"],
                ax=ax4, cbar=False, linewidths=0.5)
    ax4.set_title(f"Conf. Matrix (Acc={res_clf['acc']:.3f})")
    ax4.set_facecolor(PALETTE["fundo"])

    ax5 = fig.add_subplot(gs[1, 2])
    fpr, tpr, _ = roc_curve(res_clf["y_test"], res_clf["pred_proba"])
    ax5.plot(fpr, tpr, color=PALETTE["chuva"], lw=2, label=f"AUC={res_clf['auc']:.3f}")
    ax5.plot([0,1],[0,1],"--", color=PALETTE["grid"], lw=1)
    ax5.set_title("Curva ROC — Chuva"); ax5.legend(fontsize=8)
    ax5.set_facecolor(PALETTE["fundo"])

    fig.tight_layout()
    path = f"{OUTPUT_DIR}/11_painel_ml.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Gráfico] Salvo: {path}")


def exportar_metricas(res_reg, res_clf):
    linhas = []
    for m in res_reg["metricas"]:
        linhas.append({"Modelo": m["nome"], "Tarefa": "Regressão (Temp. T+1)",
                        "MAE (°C)": round(m["MAE"],3), "RMSE (°C)": round(m["RMSE"],3),
                        "R²": round(m["R2"],4), "Acurácia": "-", "AUC-ROC": "-"})
    linhas.append({"Modelo": "RF Classifier", "Tarefa": "Classificação (Chuva T+1)",
                    "MAE (°C)": "-", "RMSE (°C)": "-", "R²": "-",
                    "Acurácia": round(res_clf["acc"],4), "AUC-ROC": round(res_clf["auc"],4)})
    df_m = pd.DataFrame(linhas)
    path = f"{OUTPUT_DIR}/metricas_ml.csv"
    df_m.to_csv(path, index=False)
    print(f"\n[Export] Métricas salvas em {path}")
    print(df_m.to_string(index=False))


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 52)
    print("  ClimaSat — Step 3: Machine Learning")
    print("=" * 52)

    df = carregar_e_preparar(INPUT_CSV)
    feat_reg, feat_clf = separar_features(df)

    print("\n── Treinando modelos de regressão ───────────────")
    res_reg = treinar_regressao(df, feat_reg)

    print("\n── Treinando classificador de chuva ─────────────")
    res_clf = treinar_classificacao(df, feat_clf)

    print("\n── Gerando visualizações ─────────────────────────")
    grafico_previsto_real(res_reg)
    grafico_serie_predicao(res_reg)
    grafico_residuos(res_reg)
    grafico_importancia(res_reg["importancias"],
        "Importância de features — RF Regressor (temperatura)", "09_feat_imp_reg")
    grafico_classificacao(res_clf)
    grafico_importancia(res_clf["importancias"],
        "Importância de features — RF Classificador (chuva)", "09b_feat_imp_clf")
    grafico_painel_ml(res_reg, res_clf)

    exportar_metricas(res_reg, res_clf)

    print(f"\n[✓] Step 3 concluído! Arquivos em: {OUTPUT_DIR}")
