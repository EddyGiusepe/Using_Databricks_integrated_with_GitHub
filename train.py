#! /usr/bin/env python
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script train.py
===============
Neste script, vamos treinar um modelo de machine learning para prever se um
cliente vai cancelar seu serviço de telecomunicações. Para isso, vamos usar o
dataset Churn_Modelling.csv, que contém  informações sobre clientes de uma
empresa de telecomunicações. Ademais, vamos a implementar um pipeline de
machine learning para treinar o modelo.

RUN
---
uv run train.py

NOTA:
Talvez necessite instalar:
* sudo apt-get update
* sudo apt-get install git-lfs 
* git lfs install   (configuração no meu sistema)

TAMBÉM:
* git lfs track Churn_Modelling.csv (Isto serve para o git lfs saber que o arquivo é grande e deve ser armazenado em um servidor remoto.)
"""
import matplotlib.pyplot as plt
import pandas as pd
import skops.io as sio # É uma biblioteca para salvar e carregar modelos de machine learning.
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression


class ChurnModelPipeline:
    """Pipeline para treinamento de modelo de previsão de churn (cancelamento de contrato) em telecomunicações.

    Esta classe implementa um pipeline completo para treinamento de modelo de machine learning,
    incluindo carregamento de dados, pré-processamento, seleção de características e treinamento
    do modelo final.

    Attributes:
        data_path (str): Caminho para o arquivo CSV com os dados
        target_column (str): Nome da coluna alvo/target para previsão
        index_col (str | int | None): Coluna(s) a ser usada como índice do DataFrame
        random_state (int): Semente aleatória para reprodutibilidade
        model_pipeline (Pipeline | None): Pipeline completo do modelo após treinamento
        X_train (pd.DataFrame | None): Features de treino
        X_test (pd.DataFrame | None): Features de teste  
        y_train (pd.Series | None): Target de treino
        y_test (pd.Series | None): Target de teste
        preprocessor (ColumnTransformer | None): Transformador para pré-processamento dos dados
    """
    def __init__(self, data_path, target_column, index_col=None, random_state=125):
        """Inicializa a classe do pipeline do modelo com o dataset e a configuração."""
        self.data_path = data_path
        self.target_column = target_column
        self.index_col = index_col
        self.random_state = random_state
        self.model_pipeline = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.preprocessor = None

    def load_and_prepare_data(self, drop_columns=None, nrows=None):
        """Carrega e embaralha os dados, então divide em features e target."""
        print("Carregando dados...")
        df = pd.read_csv(self.data_path, index_col=self.index_col, nrows=nrows)
        if drop_columns:
            df = df.drop(drop_columns, axis=1)
        df = df.sample(frac=1)  # Embaralha os dados
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        print("Dados carregados e divididos com sucesso.")

    def build_preprocessor(self, cat_cols, num_cols):
        """Cria o pipeline de pré-processamento para dados numéricos e categóricos."""
        print("Criando pipeline de pré-processamento...")
        numerical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())] # Preenche os valores faltantes com a média e escala os dados.
        )
        categorical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder())] # Preenche os valores faltantes com a moda e codifica os dados categóricos.
        )
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, num_cols), # Transforma os dados numéricos.
                ("cat", categorical_transformer, cat_cols), # Transforma os dados categóricos.
            ],
            remainder="passthrough", # Mantém as colunas que não foram transformadas.
        )
        print("Pipeline de pré-processamento criada com sucesso.")

    def build_model_pipeline(self, k_best=5):
        """Cria o pipeline completo do modelo, incluindo seleção de características e classificador."""
        print("Criando pipeline do modelo...")
        feature_selector = SelectFromModel(LogisticRegression(max_iter=1000)) # Seleciona as características mais importantes.
        model = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state) # Cria o modelo de Gradient Boosting.
        
        train_pipeline = Pipeline(steps=[("feature_selection", feature_selector), ("GBmodel", model)]) # Cria o pipeline de treinamento.

        self.model_pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("train", train_pipeline),
            ]
        )
        print("Pipeline do modelo criada com sucesso.")

    def train_model(self):
        """Treina o modelo nos dados de treinamento."""
        if self.model_pipeline is None:
            raise ValueError("Pipeline do modelo não inicializada. Construa o pipeline do modelo primeiro.")
        
        print("Treinando o modelo...")
        self.model_pipeline.fit(self.X_train, self.y_train)
        print("Treinamento do modelo concluído.")

    def evaluate_model(self):
        """Avalia o modelo nos dados de teste e imprime as principais métricas."""
        print("Avaliando o modelo...")
        predictions = self.model_pipeline.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average="macro")
        print(f"Acurácia: {round(accuracy * 100, 2)}%, F1 Score: {round(f1, 2)}")
        
        return accuracy, f1

    def plot_confusion_matrix(self):
        """Plota e salva a matriz de confusão."""
        print("Plotando matriz de confusão...")
        predictions = self.model_pipeline.predict(self.X_test)
        cm = confusion_matrix(self.y_test, predictions, labels=self.model_pipeline.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model_pipeline.classes_)
        disp.plot()
        plt.savefig("model_results.png", dpi=120)
        print("Matriz de confusão salva como 'model_results.png'.")

    def save_metrics(self, accuracy, f1):
        """Salva as métricas de avaliação em um arquivo de texto."""
        print("Salvando métricas em um arquivo de texto...")
        with open("metrics.txt", "w") as outfile:
            outfile.write(f"Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}\n")
        print("Métricas salvas em 'metrics.txt'.")

    def plot_roc_curve(self):
        """Plota e salva a curva ROC para o classificador."""
        print("Plotando curva ROC...")
        y_probs = self.model_pipeline.predict_proba(self.X_test)[:, 1]  # Probabilidades para a classe 1
        fpr, tpr, _ = roc_curve(self.y_test, y_probs)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(self.y_test, y_probs):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falso Positivo')
        plt.ylabel('Taxa de Verdadeiro Positivo')
        plt.title('Curva ROC (Receiver Operating Characteristic)')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png", dpi=120) # dpi significa "dots per inch" e é a resolução da imagem.
        print("Curva ROC salva como 'roc_curve.png'.")


    def save_pipeline(self):
        """Salva o pipeline treinado usando skops."""
        print("Salvando pipeline em um arquivo...")
        sio.dump(self.model_pipeline, "churn_pipeline.skops")
        print("Pipeline salva como 'churn_pipeline.skops'.")


if __name__ == "__main__":
    # Configuração:
    data_file = "Churn_Modelling.csv"  # Caminho para o dataset
    target_col = "Exited"  # Coluna alvo
    drop_cols = ["RowNumber", "CustomerId", "Surname"]  # Colunas a serem removidas
    # Atualizados indices após remover 'RowNumber', 'CustomerId', 'Surname'
    cat_columns = [1, 2]  # 'Geography', 'Gender' após remover colunas
    num_columns = [0, 3, 4, 5, 6, 7, 8, 9]  # Colunas numéricas restantes
    
    # Inicializa e constrói o pipeline:
    churn_pipeline = ChurnModelPipeline(data_file, target_col)
    churn_pipeline.load_and_prepare_data(drop_columns=drop_cols, nrows=1000)
    churn_pipeline.build_preprocessor(cat_cols=cat_columns, num_cols=num_columns)
    churn_pipeline.build_model_pipeline()

    # Treina e avalia o modelo:
    churn_pipeline.train_model()
    accuracy, f1 = churn_pipeline.evaluate_model()

    # Plota matriz de confusão e salva métricas:
    churn_pipeline.plot_confusion_matrix()
    churn_pipeline.save_metrics(accuracy, f1)
    churn_pipeline.plot_roc_curve()

    # Salva o pipeline:
    churn_pipeline.save_pipeline()
    print("CI/CD concluído com sucesso. Dr. Eddy Giusepe")
