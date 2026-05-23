# Resumen del Análisis y Modelado de Ventas de Videojuegos

En este pryecyo se aborda el análisis y modelado de un dataset de ventas globales de videojuegos, cubriendo la carga de datos, EDA, modelado de regresión, clasificación y clustering.

## 1. Carga y Análisis Exploratorio de Datos (EDA)

El Análisis Exploratorio de Datos (EDA) del dataset `cleaned_global_video_game_sales.csv` reveló una estructura de `11470 filas y 14 columnas` sin valores nulos. Se observó que las `Global_Sales` tenían una distribución asimétrica, con predominio de juegos con bajas ventas. El análisis por `año` identificó períodos de mayor lanzamiento y las `plataformas` principales fueron `PSTow`, `DS` y `PS`. Finalmente, la visualización de `ventas por género` permitió entender la rentabilidad de cada categoría, ofreciendo una visión integral de las tendencias del mercado en el conjunto de datos.

## 2. Modelos de Regresión (Predicción de `Global_Sales`)

Se probaron varios modelos de regresión:

*   **RandomForestRegressor**: Modelos iniciales (`MAE: 0.056`, `R2: 0.811`).
*   **GradientBoostingRegressor**: El mejor desempeño después de `GridSearchCV` (`MAE: 0.024`, `R2: 0.854`). Logró un alto `R2` en validación cruzada (`0.994`).
*   **Support Vector Regressor (SVR)**: Tuvo un rendimiento moderado (`MAE: 0.689`, `R2: 0.556`).
*   **LinearRegression**: La optimización con `GridSearchCV` fue interrumpida.

## 3. Modelos de Clasificación (Predicción de "Exitoso")

Se entrenaron modelos para clasificar si un juego es "Exitoso", utilizando `SMOTE` para el balanceo de clases:

*   **DecisionTreeClassifier**: `Accuracy: 0.761`, `F1 Score: 0.518`.
*   **RandomForestClassifier**: Mejoró el rendimiento (`Accuracy: 0.771`, `F1 Score: 0.569`).
*   **LogisticRegression**: Destacó por su `Recall` (`0.736`) para identificar juegos exitosos, con `Accuracy: 0.735` y `F1 Score: 0.581`.

## 4. Modelos No Supervisados (Clustering)

Se exploraron métodos para identificar patrones de agrupamiento:

*   **K-Means**: Generó 4 clusters con un `Silhouette Score: 0.423`, pero con distribución desbalanceada.
*   **DBSCAN**: Identificó 183 clusters y mucho ruido (`8524 registros`), pero logró un `Silhouette Score` alto en los clusters válidos (`0.687`), indicando buena separación.
*   **Agglomerative Clustering**: Obtuvo un `Silhouette Score: 0.390`, con clusters muy desbalanceados.
*   **Gaussian Mixture Model (GMM)**: Presentó un `Silhouette Score` negativo (`-0.008`), indicando un agrupamiento deficiente.

## Conclusiones Clave

*   Para **Regresión**, el **GradientBoostingRegressor** fue el modelo más eficaz.
*   Para **Clasificación**, **RandomForestClassifier** y **LogisticRegression** ofrecieron buen rendimiento, el segundo destacando en `Recall`.
*   En **Clustering**, **DBSCAN** mostró la mejor separación entre los grupos identificados, a pesar de la presencia de ruido.
# Estructura del proyecto

modelos-machine-learning/
│
├── conf/ # Configuración de Kedro
├── data/ # Datos del proyecto
│ ├── 01_raw/
│ ├── 02_intermediate/
│ ├── 03_primary/
│ ├── 04_feature/
│ ├── 05_model_input/
│ ├── 06_models/
│ └── 07_model_output/
│
├── notebooks/
│ ├── 01_exploratory_analysis.ipynb
│ ├── 02_supervised_modeling.ipynb
│ ├── 03_model_evaluation.ipynb
│ ├── 04_hyperparameter_optimization.ipynb
│ └── 05_final_analysis.ipynb
│
├── src/
│ └── modelos_machine_learning/
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ └── hyperparameter_tuning.py
│
├── models/
│ └── trained_models/
│
├── results/
│ ├── metrics/
│ ├── plots/
│ └── reports/
│
├── requirements.txt
└── README.md

