import numpy as np  # biblioteca para operações matemáticas e arrays
import pandas as pd    # biblioteca para manipulação de dados em tabelas
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset ='game_dataset.csv'
df = pd.read_csv('game_dataset.csv')

# TIPOS DE DADOS
print("\n\n • TIPOS DE DADOS •\n")
print("\n\nTipos antes:\n", df.dtypes)

# Remoção de colunas desnecessárias
# Remove colunas irrelevantes para análise - sem nome:
colunas_irrelevantes = ['Unnamed: 0']
colunas_existentes = [c for c in colunas_irrelevantes if c in df.columns]

df = df.drop(columns=colunas_existentes)
print(f"• Colunas removidas: {colunas_existentes}")
print(f"• Colunas restantes: {df.columns.tolist()}") #Transforma os nomes das colunas em uma lista comum no python

# Remove as aspas extras de todos os nomes de colunas
df.columns = [col.replace("'", "") for col in df.columns]

# DIMENSÕES
print("\n\n • DIMENSÕES •\n")
print('• Quantidade de linhas e colunas desse dataset:')
print(df.shape, '\n\n') 

# VALORES AUSENTES
print("\n\n • VALORES AUSENTES •\n")
nulos = df.isnull().sum() # Quantifica nulos por coluna
print("• Nulos por coluna:", nulos)

if nulos.empty:
    #Preenchendo classificações faltantes com a mediana
    mediana_classe = df['Classificação de Usuários'].median()
    print (f"A média das notas é: {mediana_classe}")
    df['Classificação de Usuários'].fillna(mediana_classe, inplace=True)


    # Confirma que não restam nulos
    print("• Nulos após tratamento:\n", df.isnull().sum())

# DUPLICATAS
print("\n\n • DUPLICATAS •\n")
qtd_dup = df.duplicated().sum()
if qtd_dup>0:
    print(f"• Registros duplicados encontrados: {qtd_dup}")

    # Remove duplicatas, mantendo a primeira ocorrência
    df = df.drop_duplicates(keep='first')
    df = df.reset_index(drop=True)  # reorganiza o índice após remoções

    print(f"• Shape final do DataFrame: {df.shape}")


# TRANSFORMAÇÃO DE VARIÁVEIS CATEGÓRICAS
print("\n\n • TRANSFORMAÇÃO DE VARIÁVEIS CATEGÓRICAS •\n")

# --- 1. Label Encoding: 'Lançamento no Brasil' (binária: Sim/Não) ---
# Mapeia 'Sim' → 1 e 'Não' → 0 diretamente, sem dependência de bibliotecas externas.
df['Lançamento no Brasil'] = df['Lançamento no Brasil'].map({'Sim': 1, 'Não': 0})
print("• Label Encoding aplicado em 'Lançamento no Brasil'")
print("  Valores únicos resultantes:", df['Lançamento no Brasil'].unique())

# --- 2. One-Hot Encoding: 'Gênero' (5 categorias sem ordem) ---
# pd.get_dummies cria uma coluna binária para cada categoria.
# drop_first=False mantém todas as colunas para facilitar a interpretação.
# dtype=int converte os booleanos gerados para 0/1 inteiros.
df = pd.get_dummies(df, columns=['Gênero'], drop_first=False, dtype=int)
print("\n• One-Hot Encoding aplicado em 'Gênero'")
print("  Novas colunas:", [c for c in df.columns if c.startswith('Gênero')])

# --- 3. One-Hot Encoding: 'Plataforma' (5 categorias sem ordem) ---
df = pd.get_dummies(df, columns=['Plataforma'], drop_first=False, dtype=int)
print("\n• One-Hot Encoding aplicado em 'Plataforma'")
print("  Novas colunas:", [c for c in df.columns if c.startswith('Plataforma')])

# Resultado final da etapa de encoding
print("\n• Colunas após todas as transformações:")
print(df.columns.tolist())
print("\n• Tipos de dados finais:\n", df.dtypes)
print("\n• Primeiras linhas do DataFrame transformado:\n")
print(df.head())


# DIVISÃO ENTRE TREINO E TESTE
print("\n\n • DIVISÃO TREINO E TESTE •\n")

# Remove a coluna 'Nome do Jogo' — é um identificador textual, não uma feature
df = df.drop(columns=['Nome do Jogo'])

# Define X (features) e y (variável alvo: Preço)
X = df.drop(columns=['Preço'])
y = df['Preço']

# Divide em 80% treino e 20% teste
# random_state=42 garante reprodutibilidade (mesmos índices em toda execução)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"• Total de amostras: {len(df)}")
print(f"• Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/len(df)*100:.0f}%)")
print(f"• Teste:  {X_test.shape[0]} amostras ({X_test.shape[0]/len(df)*100:.0f}%)")
print(f"• Features (X): {X_train.shape[1]} colunas")


# NORMALIZAÇÃO DE DADOS

print("\n\n • NORMALIZAÇÃO DE DADOS •\n")

# Colunas contínuas com escalas distintas que serão normalizadas
# (excluindo binárias 0/1 que já estão na mesma escala)
colunas_normalizar = ['Classificação de Usuários', 'Vendas Globais', 'Idade Recomendada']

print("• Ranges ANTES da normalização (conjunto de treino):")
for col in colunas_normalizar:
    print(f"  {col:35s} → min: {X_train[col].min():.2f}  |  max: {X_train[col].max():.2f}")

# Instancia o scaler e ajusta APENAS no treino
scaler = MinMaxScaler()
scaler.fit(X_train[colunas_normalizar])

# Aplica a transformação no treino e no teste com os mesmos parâmetros
X_train[colunas_normalizar] = scaler.transform(X_train[colunas_normalizar])
X_test[colunas_normalizar]  = scaler.transform(X_test[colunas_normalizar])

print("\n• Ranges APÓS normalização (conjunto de treino):")
for col in colunas_normalizar:
    print(f"  {col:35s} → min: {X_train[col].min():.4f}  |  max: {X_train[col].max():.4f}")

print("\n• Ranges APÓS normalização (conjunto de teste):")
for col in colunas_normalizar:
    print(f"  {col:35s} → min: {X_test[col].min():.4f}  |  max: {X_test[col].max():.4f}")

print("\n• Primeiras linhas de X_train após normalização:\n")
print(X_train.head())

# RESUMO FINAL — DATASET PRONTO PARA MODELAGEM

print("\n\n • RESUMO FINAL DO PRÉ-PROCESSAMENTO •\n")
print(f"  ✔ Duplicatas removidas")
print(f"  ✔ Variáveis categóricas convertidas (Label + One-Hot Encoding)")
print(f"  ✔ Divisão treino/teste: 80% / 20%  (random_state=42)")
print(f"  ✔ Normalização Min-Max aplicada nas features contínuas (fit apenas no treino)")
print(f"\n  X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}  |  y_test:  {y_test.shape}")
print(f"\n  Features finais: {X_train.columns.tolist()}")
 
 
 
# DESAFIO — MODELAGEM: PREVISÃO DE PREÇO COM MACHINE LEARNING

 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
 
print("\n\n •  DESAFIO — MODELAGEM DE MACHINE LEARNING •\n")
 
# Os modelos do scikit-learn não aceitam NaN — preenchemos com a mediana do treino
# para não vazar informação do teste nessa etapa também.
mediana_treino = X_train.median()
X_train_modelo = X_train.fillna(mediana_treino)
X_test_modelo  = X_test.fillna(mediana_treino)  # usa mediana do TREINO, não do teste
 
 
# --- MODELO 1: Regressão Linear (baseline) ---
# Escolha: primeiro modelo a ser testado em qualquer regressão. Serve como
# referência mínima de desempenho — se não bater o baseline, algo está errado.
print("• Modelo 1: Regressão Linear (baseline)\n")
modelo_lr = LinearRegression()
modelo_lr.fit(X_train_modelo, y_train)
pred_lr = modelo_lr.predict(X_test_modelo)
 
mae_lr = mean_absolute_error(y_test, pred_lr)
r2_lr  = r2_score(y_test, pred_lr)
print(f"  MAE  : R$ {mae_lr:.2f}")
print(f"  R²   : {r2_lr:.4f}")
 
 
# --- MODELO 2: Random Forest Regressor ---
# Escolha: captura relações não-lineares que a Regressão Linear ignora.
# n_estimators=100 → 100 árvores de decisão combinadas por média.
# random_state=42  → reprodutibilidade dos resultados.
print("\n• Modelo 2: Random Forest Regressor\n")
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train_modelo, y_train)
pred_rf = modelo_rf.predict(X_test_modelo)
 
mae_rf = mean_absolute_error(y_test, pred_rf)
r2_rf  = r2_score(y_test, pred_rf)
print(f"  MAE  : R$ {mae_rf:.2f}")
print(f"  R²   : {r2_rf:.4f}") # tipo mediana
 
 
# --- COMPARATIVO E ANÁLISE DOS RESULTADOS ---
print("\n• Comparativo de desempenho:")
print(f"  {'Modelo':<28} {'MAE':>10}  {'R²':>8}")
print(f"  {'-'*50}")
print(f"  {'Regressão Linear':<28} {'R$ '+str(round(mae_lr,2)):>10}  {r2_lr:>8.4f}")
print(f"  {'Random Forest':<28} {'R$ '+str(round(mae_rf,2)):>10}  {r2_rf:>8.4f}")
 
print("""
• O que foi tentado:
  Dois algoritmos clássicos de regressão foram testados: Regressão Linear
  como baseline e Random Forest para capturar possíveis não-linearidades.
 
• O que não funcionou:
  Ambos os modelos apresentaram R² negativo, o que significa que nenhum deles
  conseguiu explicar a variância do preço melhor do que simplesmente prever
  a média para todos os jogos. Isso indica que as features disponíveis
  (gênero, plataforma, vendas, classificação, idade recomendada) não possuem
  correlação estatisticamente relevante com o preço neste dataset.
 
• Por que isso aconteceu:
  Datasets gerados sinteticamente (como este, com nomes "Jogo_495") muitas
  vezes atribuem o preço de forma aleatória, sem dependência real das outras
  variáveis. Em dados reais de jogos, seria esperado que plataforma, gênero
  e vendas influenciassem o preço — aqui, essa relação não existe nos dados.
 
• O que funcionaria melhor em dados reais:
  Com dados reais e correlações genuínas, Gradient Boosting (XGBoost, LightGBM)
  seria o próximo passo natural, por sua capacidade de capturar padrões
  complexos e lidar bem com features mistas (contínuas + binárias).
""")
