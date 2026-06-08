#  COMPARAÇÃO DE ALGORITMOS DE REGRESSÃO
#  Dataset: Salary Dataset (YearsExperience → Salary)
#
#  Objetivos cobertos:
#  - Tipos de regressão: Linear, Polinomial, Random Forest
#  - Pré-processamento: remoção de colunas desnecessárias,
#    verificação de nulos, separação treino/teste
#  - Métricas: RMSE e MAE para cada modelo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. COLETA DE DADOS
#   Dataset: YearsExperience (anos de experiência) → Salary (salário)
#   30 registros reais de funcionários

df = pd.read_csv("Salary_dataset.csv")

# Remove coluna de índice desnecessária
df.drop(columns=["Unnamed: 0"], inplace=True)

print("=== Primeiras linhas do dataset ===")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Colunas: {list(df.columns)}")

# 2. PRÉ-PROCESSAMENTO


# 2a. Verificação de elementos faltantes
print(f"\nValores faltantes por coluna:\n{df.isnull().sum()}")
# Não há valores faltantes neste dataset

# 2b. Variáveis categóricas
# Não há variáveis categóricas — ambas as colunas são numéricas

# 2c. Normalização
# Não aplicada: os dados já estão em escala adequada para regressão
# e os modelos escolhidos não exigem normalização

# 2d. Separação: variável independente X e dependente y
X = df[["YearsExperience"]].values   # feature: anos de experiência
y = df["Salary"].values              # alvo: salário

# Divisão treino/teste — 70% treino, 30% teste
# random_state=42 garante reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
print(f"\nAmostras de treino: {len(X_train)}")
print(f"Amostras de teste : {len(X_test)}")

# 3. IMPLEMENTAÇÃO E TREINO DOS MODELOS

# --- MODELO 1: Regressão Linear Simples ---
# Ajusta uma reta y = a*x + b minimizando o erro quadrático
# Ideal quando a relação entre variáveis é aproximadamente linear
m_linear = LinearRegression()
m_linear.fit(X_train, y_train)
print(f"\n[Linear] Coeficiente: {m_linear.coef_[0]:.2f} | Intercepto: {m_linear.intercept_:.2f}")

# --- MODELO 2: Regressão Polinomial (grau 3) ---
# Expande as features: [x] → [1, x, x², x³]
# Captura relações não-lineares entre as variáveis
poly = PolynomialFeatures(degree=3)
X_train_p = poly.fit_transform(X_train)  # transforma treino
X_test_p  = poly.transform(X_test)       # aplica a mesma transformação no teste
m_poly = LinearRegression()
m_poly.fit(X_train_p, y_train)
print(f"[Polinomial] Coeficientes: {np.round(m_poly.coef_, 2)}")

# --- MODELO 3: Random Forest Regressor ---
# Ensemble de árvores de decisão — robusto e não-linear
# n_estimators=100: usa 100 árvores para melhor estabilidade
m_rf = RandomForestRegressor(n_estimators=100, random_state=42)
m_rf.fit(X_train, y_train)
print(f"[Random Forest] Árvores treinadas: {m_rf.n_estimators}")

# 4. AVALIAÇÃO DOS MODELOS
#    RMSE (Root Mean Square Error): penaliza erros grandes
#    MAE  (Mean Absolute Error):    erro médio absoluto, mais intuitivo

pred_linear = m_linear.predict(X_test)
pred_poly   = m_poly.predict(X_test_p)
pred_rf     = m_rf.predict(X_test)

def calcular_metricas(nome, y_real, y_pred):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae  = mean_absolute_error(y_real, y_pred)
    print(f"  {nome:<25} RMSE: {rmse:>10.2f}  |  MAE: {mae:>10.2f}")
    return rmse, mae

print("\n=== Métricas de Avaliação (conjunto de teste) ===")
r1, a1 = calcular_metricas("Regressão Linear",      y_test, pred_linear)
r2, a2 = calcular_metricas("Regressão Polinomial",  y_test, pred_poly)
r3, a3 = calcular_metricas("Random Forest",         y_test, pred_rf)

# Identifica o melhor modelo
metricas = {"Regressão Linear": r1, "Regressão Polinomial": r2, "Random Forest": r3}
melhor = min(metricas, key=metricas.get)
print(f"\n>>> Melhor modelo pelo RMSE: {melhor} (RMSE = {metricas[melhor]:.2f})")

# 5. VISUALIZAÇÃO

# Reta/curva contínua para plotar cada modelo
x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_linear = m_linear.predict(x_range)
y_poly   = m_poly.predict(poly.transform(x_range))
y_rf     = m_rf.predict(x_range)

# --- Gráfico 1: curvas dos 3 modelos ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Comparação de Modelos — Anos de Experiência vs. Salário",
             fontsize=13, fontweight="bold")

configs = [
    ("Regressão Linear",    pred_linear, y_linear, "#E74C3C", r1, a1),
    ("Regressão Polinomial",pred_poly,   y_poly,   "#2ECC71", r2, a2),
    ("Random Forest",       pred_rf,     y_rf,     "#9B59B6", r3, a3),
]

for ax, (titulo, pred, curva, cor, rmse, mae) in zip(axes, configs):
    # Pontos do conjunto completo (cinza claro) + teste (azul)
    ax.scatter(X_train, y_train, color="#AAAAAA", alpha=0.5, s=50, label="Treino")
    ax.scatter(X_test,  y_test,  color="#4C72B0", alpha=0.8, s=60, edgecolors="white",
               linewidth=0.5, label="Teste (real)")
    ax.plot(x_range, curva, color=cor, linewidth=2.5, label="Modelo")
    ax.set_title(f"{titulo}\nRMSE = {rmse:,.0f}  |  MAE = {mae:,.0f}", fontsize=10)
    ax.set_xlabel("Anos de Experiência")
    ax.legend(fontsize=8)

axes[0].set_ylabel("Salário (USD)")
plt.tight_layout()
plt.savefig("grafico_modelos.png", dpi=150)
print("\nGráfico 1 salvo: grafico_modelos.png")

# --- Gráfico 2: comparação de métricas (barras) ---
modelos_nomes = ["Linear", "Polinomial", "Random Forest"]
rmses = [r1, r2, r3]
maes  = [a1, a2, a3]

x = np.arange(len(modelos_nomes))
width = 0.35

fig2, ax2 = plt.subplots(figsize=(8, 5))
bars1 = ax2.bar(x - width/2, rmses, width, label="RMSE", color="#E74C3C", alpha=0.85)
bars2 = ax2.bar(x + width/2, maes,  width, label="MAE",  color="#3498DB", alpha=0.85)

for bar in list(bars1) + list(bars2):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 200,
             f"{bar.get_height():,.0f}",
             ha="center", va="bottom", fontsize=9)

ax2.set_xticks(x)
ax2.set_xticklabels(modelos_nomes, fontsize=11)
ax2.set_ylabel("Erro (USD)")
ax2.set_title("RMSE e MAE por Modelo", fontsize=12, fontweight="bold")
ax2.legend(fontsize=11)
ax2.set_ylim(0, max(rmses) * 1.3)
plt.tight_layout()
plt.savefig("grafico_metricas.png", dpi=150)
print("Gráfico 2 salvo: grafico_metricas.png")
ax2.set_xticklabels(modelos_nomes, fontsize=11)
ax2.set_ylabel("Erro (USD)")
ax2.set_title("RMSE e MAE por Modelo", fontsize=12, fontweight="bold")
ax2.legend(fontsize=11)
ax2.set_ylim(0, max(rmses) * 1.3)
plt.tight_layout()
plt.savefig("grafico_metricas.png", dpi=150)
print("Gráfico 2 salvo: grafico_metricas.png")