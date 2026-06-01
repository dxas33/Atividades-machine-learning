# ============================================================
# Regressão Linear Simples — Previsão de Salário
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------
# 1. COLETA DE DADOS
# ----------------------------------------------------------
df = pd.read_csv("Salary_dataset.csv")   # ← Alterado aqui

print("=== Primeiras linhas do dataset ===")
print(df.head())
print(f"\nShape: {df.shape}")

# ----------------------------------------------------------
# 2. PRÉ-PROCESSAMENTO
# ----------------------------------------------------------
X = df[["YearsExperience"]].values
y = df["Salary"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ----------------------------------------------------------
# 3. MODELAGEM
# ----------------------------------------------------------
modelo = LinearRegression()
modelo.fit(X_train, y_train)

print(f"\n=== Modelo treinado ===")
print(f"Coeficiente: {modelo.coef_[0]:.2f}")
print(f"Intercepto : {modelo.intercept_:.2f}")

# ----------------------------------------------------------
# 4. AVALIAÇÃO
# ----------------------------------------------------------
y_pred = modelo.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n=== Avaliação ===")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f} ({r2*100:.1f}%)")

# ----------------------------------------------------------
# 5. VISUALIZAÇÃO
# ----------------------------------------------------------
ordem = np.argsort(X_test[:, 0])
x_sorted = X_test[ordem, 0]
y_sorted = modelo.predict(x_sorted.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="#4C72B0", alpha=0.7, s=70, label="Dados reais")
plt.plot(x_sorted, y_sorted, color="#DD4444", linewidth=3, label="Regressão Linear")
plt.xlabel("Anos de Experiência")
plt.ylabel("Salário (US$)")
plt.title("Regressão Linear - Salário vs Experiência")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_regressao_salario.png", dpi=200)
print("\n✅ Gráfico salvo como: grafico_regressao_salario.png")