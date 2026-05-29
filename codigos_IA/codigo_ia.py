import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score

# ── 1. Carregamento ──────────────────────────────────────────────────────────
df = pd.read_csv('f1_strategy_dataset_v4.csv')

# ── 2. Pré-processamento ─────────────────────────────────────────────────────

# Preenche o único nulo: Compound (66 linhas) com a moda
df['Compound'] = df['Compound'].fillna(df['Compound'].mode()[0])

# Encoding das colunas categóricas (texto → número)
le = LabelEncoder()
for col in ['Driver', 'Compound', 'Race']:
    df[col] = le.fit_transform(df[col])

# Normaliza as colunas com escala muito diferente
cols_to_normalize = ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'TyreLife']
scaler = MinMaxScaler()
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# ── 3. Features e Target ─────────────────────────────────────────────────────
X = df.drop(columns=['Position'])
y = df['Position']

print("X:", X.shape)
print("y:", y.shape)

# ── 4. Split ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# ── 5. Modelo ─────────────────────────────────────────────────────────────────
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

# ── 6. Avaliação ──────────────────────────────────────────────────────────────
acuracia = accuracy_score(y_test, y_pred)
print(f"\nAcurácia: {acuracia:.2f}")
print(f"Acurácia em porcentagem: {acuracia * 100:.2f}%")