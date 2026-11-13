# Machine-learning-ABP

Trabalho final de Machine Learning. **Dataset 09 — Wine Quality** (vinhos tintos e brancos).

## Parte 1 — Regressão (predição da nota de qualidade)

**Objetivo:** prever a nota `quality` do vinho como um valor contínuo, a partir das variáveis físico-químicas.

| Etapa | O que foi feito | Onde ver no notebook |
|---|---|---|
| 1. Análise exploratória | Histogramas, matriz de correlação, scatter plots (incluindo as 4 variáveis mais correlacionadas), nota por tipo de vinho e correlação por tipo | Seção 1 |
| 2. Limpeza e pré-processamento | Correção de escalas e erros do CSV, valores ausentes (mediana), outliers (IQR, boxplots e teste de sensibilidade), codificação de `type`, padronização | Seções 1.2 e 2 |
| 3. Modelagem | Regressão Linear, SVR e Random Forest | Seção 3 |
| 4. Avaliação | RMSE, MAE e R² contra um baseline, treino x teste, validação cruzada, real x previsto, resíduos e erro por nota | Seção 4 |
| 5. Ajuste de hiperparâmetros | `GridSearchCV` (Ridge e SVR) e `RandomizedSearchCV` (Random Forest), com comparação antes x depois | Seção 5 |
| 6. Discussão | Coeficientes, VIF, importância da Random Forest, importância por permutação, resumo e limitações | Seção 6 |

## Estrutura

```
Parte 1/
├── Parte1_Regressao.ipynb   # notebook com o código e a explicação de cada saída
└── parte1_regressao.py      # mesmo código, em formato de script
```

## Como rodar

1. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
2. Coloque o dataset em `Docs/DataSet/group_9_winequality.csv`. A pasta `Docs/` está no `.gitignore`, então o CSV **não vai para o repositório** e precisa ser colocado manualmente.
3. Abra o notebook em `Parte 1/` e use **Run All**, ou rode o script:
   ```
   cd "Parte 1"
   python parte1_regressao.py
   ```

A busca de hiperparâmetros (etapa 5) usa todos os núcleos do computador (`n_jobs=-1`) e pode levar alguns minutos.

## Observações

- Os gráficos do script `.py` abrem em janelas; feche cada uma para continuar a execução.
- Todos os sorteios usam `random_state=42`, então os resultados se repetem.
- O conjunto de teste só é usado para medir o resultado final. A escolha dos hiperparâmetros usa só o treino (validação cruzada).
