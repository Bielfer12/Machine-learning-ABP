import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pasta_do_script = os.path.dirname(os.path.abspath(__file__))
caminho_csv = os.path.join(pasta_do_script, "..", "Docs", "DataSet", "group_9_winequality.csv")

wine = pd.read_csv(caminho_csv)
print("Linhas e colunas do dataset original:", wine.shape)

antes = len(wine)
wine = wine.dropna(subset=["type"])
print("Removidas por type ausente:", antes - len(wine))

wine.loc[wine["type"].str.startswith("white"), "type"] = "white"

antes = len(wine)
wine = wine[wine["quality"] <= 10]
print("Removidas por nota inválida:", antes - len(wine))
alcohol_num = pd.to_numeric(wine["alcohol"], errors="coerce")
erros = alcohol_num.isna()


alcohol_texto = wine.loc[erros, "alcohol"].str.replace(".", "", regex=False).astype(float)

wine["alcohol"] = alcohol_num
wine.loc[erros, "alcohol"] = alcohol_texto
for i in range(40):
    wine.loc[wine["alcohol"] > 20, "alcohol"] = wine["alcohol"] / 10

for i in range(4):
    wine.loc[wine["density"] > 2, "density"] = wine["density"] / 10

antes = len(wine)
wine = wine[(wine["density"] >= 0.98) & (wine["density"] <= 1.04)]
print("Removidas por densidade impossível:", antes - len(wine))

wine.loc[wine["chlorides"] > 1, "chlorides"] = wine["chlorides"] / 1000
wine.loc[wine["volatile acidity"] > 5, "volatile acidity"] = wine["volatile acidity"] / 1000

faixas = {
    "fixed acidity": (3.5, 16),
    "volatile acidity": (0.05, 1.6),
    "citric acid": (0, 1.7),
    "residual sugar": (0.5, 66),
    "chlorides": (0.005, 0.62),
    "free sulfur dioxide": (1, 290),
    "total sulfur dioxide": (5, 440),
    "pH": (2.7, 4.1),
    "sulphates": (0.2, 2.0),
}

antes = len(wine)
for coluna, (minimo, maximo) in faixas.items():
    fora = (wine[coluna] < minimo) | (wine[coluna] > maximo)
    wine = wine[~fora]
print("Removidas por valor fora da faixa:", antes - len(wine))

antes = len(wine)
wine = wine.drop_duplicates()
print("Removidas por duplicadas:", antes - len(wine))
print("Linhas restantes após a limpeza:", len(wine))

wine.hist(bins=30, figsize=(15, 12))
plt.show()

corr_matrix = wine.corr(numeric_only=True)

print("\nCorrelação com a nota (quality):")
print(corr_matrix["quality"].sort_values(ascending=False))

plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, cmap="crest", annot=True, fmt=".1f")
plt.show()

wine.plot(kind="scatter", x="alcohol", y="quality", alpha=0.2, grid=True)
plt.show()


# --- 1.6 Scatter plots das 4 variáveis mais correlacionadas com a nota ---
# Escolhemos as 4 variáveis com maior correlação (em valor absoluto) com a nota
top4 = corr_matrix["quality"].drop("quality").abs().sort_values(ascending=False).head(4).index

fig, eixos = plt.subplots(2, 2, figsize=(12, 8))
for ax, coluna in zip(eixos.ravel(), top4):
    sns.regplot(data=wine, x=coluna, y="quality", ax=ax,
                scatter_kws={"alpha": 0.15, "s": 10}, line_kws={"color": "red"})
    ax.set_title(f"{coluna} x quality (correlação = {corr_matrix.loc[coluna, 'quality']:.2f})")
plt.tight_layout()
plt.show()

# --- 1.7 Distribuição da nota por tipo de vinho ---
# Quantos vinhos existem em cada nota, separando por tipo
print("Quantidade de vinhos por nota e por tipo:")
print(pd.crosstab(wine["quality"].astype(int), wine["type"]))

plt.figure(figsize=(8, 4))
sns.countplot(x=wine["quality"].astype(int), hue=wine["type"])
plt.xlabel("quality")
plt.title("Distribuição da nota de qualidade por tipo de vinho")
plt.show()

print("\nNota média por tipo:")
print(wine.groupby("type")["quality"].agg(["count", "mean", "std"]).round(2))

# --- 1.8 A correlação com a nota muda entre tintos e brancos? ---
# Correlação de cada variável com a nota, calculada separadamente para cada tipo
corr_por_tipo = pd.DataFrame({
    tipo: wine[wine["type"] == tipo].corr(numeric_only=True)["quality"]
    for tipo in wine["type"].unique()
}).drop("quality")
corr_por_tipo["geral"] = corr_matrix["quality"].drop("quality")

print("Correlação com a nota (quality) por tipo de vinho:")
print(corr_por_tipo.sort_values("geral", ascending=False).round(2))

ausentes = wine.isnull().sum()
print("\nValores ausentes antes de tratar:")
print(ausentes[ausentes > 0])

from sklearn.impute import SimpleImputer

wine_num = wine.select_dtypes(include=[np.number])

imputer = SimpleImputer(strategy="median")
imputer.fit(wine_num)
X = imputer.transform(wine_num)

wine_num = pd.DataFrame(X, columns=wine_num.columns, index=wine_num.index)

print("\nOutliers por coluna (IQR):")
for coluna in wine_num.columns.drop("quality"):
    q1 = wine_num[coluna].quantile(0.25)
    q3 = wine_num[coluna].quantile(0.75)
    iqr = q3 - q1
    outliers = (wine_num[coluna] < q1 - 1.5 * iqr) | (wine_num[coluna] > q3 + 1.5 * iqr)
    print(coluna, ":", outliers.sum())


# --- 2.2.1 Resumo e boxplots dos outliers ---
# Tabela com a quantidade, o percentual e os limites do IQR de cada variável
resumo = []
for coluna in wine_num.columns.drop("quality"):
    q1 = wine_num[coluna].quantile(0.25)
    q3 = wine_num[coluna].quantile(0.75)
    iqr = q3 - q1
    marcados = (wine_num[coluna] < q1 - 1.5 * iqr) | (wine_num[coluna] > q3 + 1.5 * iqr)
    resumo.append({
        "variavel": coluna,
        "outliers": int(marcados.sum()),
        "percentual (%)": round(100 * marcados.mean(), 1),
        "limite inferior": round(q1 - 1.5 * iqr, 3),
        "limite superior": round(q3 + 1.5 * iqr, 3),
    })
tabela_outliers = pd.DataFrame(resumo).sort_values("outliers", ascending=False)
print(tabela_outliers.to_string(index=False))

# Boxplot de cada variável (cada uma na sua escala)
colunas = wine_num.columns.drop("quality")
fig, eixos = plt.subplots(3, 4, figsize=(16, 10))
for ax, coluna in zip(eixos.ravel(), colunas):
    sns.boxplot(y=wine_num[coluna], ax=ax)
    ax.set_title(coluna)
    ax.set_ylabel("")
for ax in eixos.ravel()[len(colunas):]:
    ax.set_visible(False)
plt.tight_layout()
plt.show()

# --- 2.2.2 Manter ou limitar os outliers? Um teste de sensibilidade ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X_sens = wine_num.drop(columns="quality")
y_sens = wine_num["quality"]

# Versão com os outliers limitados aos limites do IQR
q1 = X_sens.quantile(0.25)
q3 = X_sens.quantile(0.75)
iqr = q3 - q1
X_limitado = X_sens.clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr, axis=1)

kf_sens = KFold(n_splits=5, shuffle=True, random_state=42)
modelos_sens = {
    "Regressão Linear": Pipeline([("scaler", StandardScaler()), ("modelo", LinearRegression())]),
    "Random Forest": Pipeline([("scaler", StandardScaler()),
                               ("modelo", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))]),
}

linhas = []
for nome, modelo in modelos_sens.items():
    rmse_original = -cross_val_score(modelo, X_sens, y_sens, cv=kf_sens,
                                     scoring="neg_root_mean_squared_error").mean()
    rmse_limitado = -cross_val_score(modelo, X_limitado, y_sens, cv=kf_sens,
                                     scoring="neg_root_mean_squared_error").mean()
    linhas.append({"modelo": nome,
                   "RMSE com outliers": rmse_original,
                   "RMSE outliers limitados": rmse_limitado,
                   "diferença": rmse_limitado - rmse_original})
print("Teste de sensibilidade aos outliers (validação cruzada, 5 partes):")
print(pd.DataFrame(linhas).round(4).to_string(index=False))

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
type_encoded = ordinal_encoder.fit_transform(wine[["type"]])

wine_num["type"] = type_encoded

X = wine_num.drop(columns="quality")
y = wine_num["quality"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTreino:", len(X_train), "linhas | Teste:", len(X_test), "linhas")

from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
X_train_scaled = std_scaler.fit_transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train_scaled, y_train)

y_pred_lin = lin_reg.predict(X_test_scaled)

from sklearn.svm import SVR

svr = SVR(kernel="rbf")

svr.fit(X_train_scaled, y_train)

y_pred_svr = svr.predict(X_test_scaled)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)

rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

print("\nPrimeiras 5 predições (nota real x modelos):")
resultado = pd.DataFrame({
    "real": y_test.values[:5],
    "linear": y_pred_lin[:5].round(2),
    "svr": y_pred_svr[:5].round(2),
    "random_forest": y_pred_rf[:5].round(2),
})
print(resultado)

# --- 4. Etapa 4 — Avaliação ---

# --- 4.1 Métricas no conjunto de teste ---
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor

def avaliar(nome, y_real, y_previsto):
    # Calcula as três métricas de um modelo
    return {
        "modelo": nome,
        "RMSE": np.sqrt(mean_squared_error(y_real, y_previsto)),
        "MAE": mean_absolute_error(y_real, y_previsto),
        "R²": r2_score(y_real, y_previsto),
    }

# Baseline: prevê sempre a média das notas do treino
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train_scaled, y_train)
y_pred_dummy = dummy.predict(X_test_scaled)

previsoes = {
    "Baseline (média)": y_pred_dummy,
    "Regressão Linear": y_pred_lin,
    "SVR": y_pred_svr,
    "Random Forest": y_pred_rf,
}

resultados_teste = pd.DataFrame(
    [avaliar(nome, y_test, pred) for nome, pred in previsoes.items()]
).set_index("modelo")
print("Métricas no conjunto de teste:")
print(resultados_teste.round(3))

melhor_modelo = resultados_teste.drop("Baseline (média)")["RMSE"].idxmin()
print(f"\nMelhor modelo no teste (menor RMSE): {melhor_modelo}")

# --- 4.2 Treino x teste: o modelo decorou os dados? ---
modelos_treinados = {"Regressão Linear": lin_reg, "SVR": svr, "Random Forest": rf}

linhas = []
for nome, modelo in modelos_treinados.items():
    r2_treino = r2_score(y_train, modelo.predict(X_train_scaled))
    r2_teste = resultados_teste.loc[nome, "R²"]
    linhas.append({"modelo": nome, "R² treino": r2_treino, "R² teste": r2_teste,
                   "diferença": r2_treino - r2_teste})
print("R² no treino x R² no teste:")
print(pd.DataFrame(linhas).set_index("modelo").round(3))

# --- 4.3 Validação cruzada ---
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

kf = KFold(n_splits=5, shuffle=True, random_state=42)

modelos_cv = {
    "Regressão Linear": Pipeline([("scaler", StandardScaler()), ("modelo", LinearRegression())]),
    "SVR": Pipeline([("scaler", StandardScaler()), ("modelo", SVR(kernel="rbf"))]),
    "Random Forest": Pipeline([("scaler", StandardScaler()),
                               ("modelo", RandomForestRegressor(random_state=42, n_jobs=-1))]),
}

linhas = []
for nome, modelo in modelos_cv.items():
    cv = cross_validate(modelo, X_train, y_train, cv=kf,
                        scoring={"rmse": "neg_root_mean_squared_error",
                                 "mae": "neg_mean_absolute_error",
                                 "r2": "r2"})
    linhas.append({
        "modelo": nome,
        "RMSE (média)": -cv["test_rmse"].mean(),
        "RMSE (desvio)": cv["test_rmse"].std(),
        "MAE (média)": -cv["test_mae"].mean(),
        "R² (média)": cv["test_r2"].mean(),
        "R² (desvio)": cv["test_r2"].std(),
    })
resultados_cv = pd.DataFrame(linhas).set_index("modelo")
print("Validação cruzada (5 partes) no conjunto de treino:")
print(resultados_cv.round(3))

# --- 4.4 Real x previsto e resíduos ---
modelos_plot = {nome: previsoes[nome] for nome in ["Regressão Linear", "SVR", "Random Forest"]}

fig, eixos = plt.subplots(2, 3, figsize=(16, 9))
for coluna, (nome, pred) in enumerate(modelos_plot.items()):
    # Linha de cima: real x previsto
    ax = eixos[0, coluna]
    ax.scatter(y_test, pred, alpha=0.25, s=12)
    ax.plot([3, 9], [3, 9], "--", color="red")
    ax.set_title(f"{nome}: real x previsto")
    ax.set_xlabel("nota real")
    ax.set_ylabel("nota prevista")

    # Linha de baixo: histograma dos resíduos
    ax = eixos[1, coluna]
    residuos = y_test.values - pred
    ax.hist(residuos, bins=30)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(f"{nome}: resíduos (média = {residuos.mean():.2f})")
    ax.set_xlabel("nota real - nota prevista")
plt.tight_layout()
plt.show()

# --- 4.5 Onde o melhor modelo erra mais? Erro por nota ---
pred_melhor = previsoes[melhor_modelo]
erro_por_nota = (
    pd.DataFrame({"nota_real": y_test.values.astype(int),
                  "erro_absoluto": np.abs(y_test.values - pred_melhor)})
    .groupby("nota_real")["erro_absoluto"]
    .agg(quantidade="count", erro_medio="mean")
)
print(f"Erro absoluto médio por nota real ({melhor_modelo}):")
print(erro_por_nota.round(3))

erro_por_nota["erro_medio"].plot(kind="bar", figsize=(7, 4))
plt.ylabel("erro absoluto médio (pontos)")
plt.title(f"Erro por nota real - {melhor_modelo}")
plt.show()

# --- 5. Etapa 5 — Ajuste de hiperparâmetros ---

# --- 5.1 Busca nos três modelos ---
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge
from scipy.stats import randint

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Ridge: GridSearchCV (poucos valores, testamos todos) ---
pipe_ridge = Pipeline([("scaler", StandardScaler()), ("modelo", Ridge())])
grade_ridge = {"modelo__alpha": [0.01, 0.1, 1, 10, 100, 1000]}
busca_ridge = GridSearchCV(pipe_ridge, grade_ridge, cv=kf,
                           scoring="neg_root_mean_squared_error", n_jobs=-1)
busca_ridge.fit(X_train, y_train)

# --- SVR: GridSearchCV (5 x 3 x 2 = 30 combinações) ---
pipe_svr = Pipeline([("scaler", StandardScaler()), ("modelo", SVR(kernel="rbf"))])
grade_svr = {
    "modelo__C": [0.5, 1, 2, 5, 10],
    "modelo__gamma": [0.02, 0.05, "scale"],
    "modelo__epsilon": [0.1, 0.2],
}
busca_svr = GridSearchCV(pipe_svr, grade_svr, cv=kf,
                         scoring="neg_root_mean_squared_error", n_jobs=-1)
busca_svr.fit(X_train, y_train)

# --- Random Forest: RandomizedSearchCV (20 combinações sorteadas) ---
pipe_rf = Pipeline([("scaler", StandardScaler()),
                    ("modelo", RandomForestRegressor(random_state=42))])
espaco_rf = {
    "modelo__n_estimators": randint(100, 500),
    "modelo__max_depth": [None, 10, 15, 20, 30],
    "modelo__min_samples_leaf": randint(1, 6),
    "modelo__max_features": ["sqrt", "log2", 0.5, 1.0],
}
busca_rf = RandomizedSearchCV(pipe_rf, espaco_rf, n_iter=20, cv=kf,
                              scoring="neg_root_mean_squared_error",
                              n_jobs=-1, random_state=42)
busca_rf.fit(X_train, y_train)

# Resumo: melhor RMSE na validação cruzada e melhores parâmetros
for nome, busca in [("Ridge", busca_ridge), ("SVR", busca_svr), ("Random Forest", busca_rf)]:
    print(f"{nome}: melhor RMSE na validação cruzada = {-busca.best_score_:.4f}")
    print("   melhores parâmetros:", busca.best_params_)

# --- 5.2 As melhores combinações do SVR ---
cv_svr = pd.DataFrame(busca_svr.cv_results_)
cv_svr["RMSE (média)"] = -cv_svr["mean_test_score"]
cv_svr["RMSE (desvio)"] = cv_svr["std_test_score"]

top5_svr = cv_svr.sort_values("rank_test_score").head(5)[
    ["param_modelo__C", "param_modelo__gamma", "param_modelo__epsilon", "RMSE (média)", "RMSE (desvio)"]
]
print("5 melhores combinações do SVR:")
print(top5_svr.round(4).to_string(index=False))

# --- 5.3 Antes x depois no conjunto de teste ---
modelos_tunados = {
    "Ridge (ajustada)": busca_ridge.best_estimator_,
    "SVR (ajustado)": busca_svr.best_estimator_,
    "Random Forest (ajustada)": busca_rf.best_estimator_,
}

previsoes_tunadas = {nome: modelo.predict(X_test) for nome, modelo in modelos_tunados.items()}
resultados_tunados = pd.DataFrame(
    [avaliar(nome, y_test, pred) for nome, pred in previsoes_tunadas.items()]
).set_index("modelo")

comparacao = pd.concat([resultados_teste, resultados_tunados])
print("Comparação final no conjunto de teste:")
print(comparacao.round(3))

# Quanto o ajuste mudou o RMSE de cada modelo
for padrao, ajustado in [("SVR", "SVR (ajustado)"), ("Random Forest", "Random Forest (ajustada)")]:
    antes = comparacao.loc[padrao, "RMSE"]
    depois = comparacao.loc[ajustado, "RMSE"]
    print(f"{padrao}: RMSE {antes:.4f} -> {depois:.4f} (variação {depois - antes:+.4f})")

comparacao["RMSE"].sort_values(ascending=False).plot(kind="barh", figsize=(8, 4))
plt.xlabel("RMSE no teste (menor é melhor)")
plt.title("Todos os modelos: RMSE no conjunto de teste")
plt.tight_layout()
plt.show()
