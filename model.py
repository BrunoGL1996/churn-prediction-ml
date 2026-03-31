import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

def treinar_modelo(df):
    # Separando o que é feature e o que é o alvo (Churn)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Separando 20% pra teste pra ver se o modelo tá aprendendo mesmo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Usei o Random Forest com 100 árvores (ajuste padrão que costuma performar bem)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Printando o resultado aqui no console pra conferir as métricas
    y_pred = rf.predict(X_test)
    print("--- Resultado do Modelo ---")
    print(classification_report(y_test, y_pred))

    return rf, X_test, y_test

def gerar_curva_roc(modelo, X_test, y_test):
    # Pegando as probabilidades pra montar a curva
    y_probs = modelo.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    score_auc = auc(fpr, tpr)

    # Plotando o gráfico que subi pro GitHub
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'AUC: {score_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--') # Linha de referência
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Desempenho do Modelo (Curva ROC)')
    plt.legend()
    plt.show()
