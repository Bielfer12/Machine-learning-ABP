# Parte 1 — Regressão

Esta pasta é uma versão separada da regressão, feita a partir da pasta `Parte 1` do repositório (EDA, pré-processamento e modelagem), sem alterar os arquivos que já existiam lá. Aqui foram acrescentados mais gráficos de EDA, a análise de outliers, as métricas, o ajuste de hiperparâmetros e a discussão.

**Objetivo:** prever a nota `quality` do vinho como um valor contínuo, a partir das variáveis físico-químicas.

| Etapa | O que tem | Seção do notebook |
|---|---|---|
| 1. Análise exploratória | Histogramas, matriz de correlação, scatter plots (incluindo as 4 variáveis mais correlacionadas), nota por tipo e correlação por tipo | 1 |
| 2. Limpeza e pré-processamento | Correção do CSV, valores ausentes, outliers (IQR, boxplots e teste de sensibilidade), codificação de `type`, padronização | 1.2 e 2 |
| 3. Modelagem | Regressão Linear, SVR e Random Forest | 3 |
| 4. Avaliação | RMSE, MAE e R² contra um baseline, treino x teste, validação cruzada, real x previsto, resíduos e erro por nota | 4 |
| 5. Hiperparâmetros | `GridSearchCV` (Ridge e SVR) e `RandomizedSearchCV` (Random Forest), antes x depois | 5 |
| 6. Discussão | Coeficientes, VIF, importâncias, resumo e limitações | 6 |

## Como rodar

1. `pip install -r requirements.txt`
2. Coloque o dataset em `Docs/DataSet/group_9_winequality.csv`, na raiz do repositório. A pasta `Docs/` está no `.gitignore`, então o CSV não vai para o GitHub.
3. Abra `Parte1_Regressao.ipynb` e use **Run All**, ou rode `python parte1_regressao.py` de dentro desta pasta.

A busca de hiperparâmetros usa todos os núcleos (`n_jobs=-1`) e pode levar alguns minutos. Os sorteios usam `random_state=42`.
