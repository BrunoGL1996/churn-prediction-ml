# ================================
# 📚 IMPORTS
# ================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import os

if not os.path.exists('outputs'):
    os.makedirs('outputs')
    print("Pasta 'outputs' criada com sucesso!")

sns.set(style="whitegrid")

def load_and_clean_data(path):
    df = pd.read_csv(path)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df.dropna(inplace=True)

    df.drop('customerID', axis=1, inplace=True)

    print("✅ Dados carregados e limpos com sucesso!")
    print(f"📊 Total de registros: {len(df)}")

    return df

def analyze_churn(df):
    print("\n--- Taxa de Churn Geral (%) ---")
    print(df['Churn'].value_counts(normalize=True) * 100)

    churn_contract = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100

    churn_payment = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
    churn_payment = churn_payment.sort_values(by='Yes', ascending=False)

    return churn_contract, churn_payment

def plot_churn_payment(churn_payment):
    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=churn_payment.index,
        y='Yes',
        data=churn_payment.reset_index(),
        palette='rocket'
    )


    for i, v in enumerate(churn_payment['Yes']):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')

    plt.title('Taxa de Churn por Meio de Pagamento', fontsize=16, fontweight='bold')
    plt.ylabel('Churn (%)')
    plt.xlabel('Método de Pagamento')

    plt.xticks(rotation=15)
    plt.ylim(0, 55)
    sns.despine()

    plt.tight_layout()
    plt.savefig('outputs/churn_pagamento.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_churn_contract(churn_contract):
    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=churn_contract.index,
        y='Yes',
        data=churn_contract.reset_index(),
        palette='viridis'
    )


    for i, v in enumerate(churn_contract['Yes']):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')

    plt.title('Taxa de Churn por Tipo de Contrato', fontsize=16, fontweight='bold')
    plt.ylabel('Churn (%)')
    plt.xlabel('Tipo de Contrato')

    plt.ylim(0, 50)
    sns.despine()

    plt.tight_layout()
    plt.savefig('outputs/churn_contrato.png', dpi=300, bbox_inches='tight')
    plt.show()



def prepare_for_ml(df):
    df_ml = df.copy()

    le = LabelEncoder()

    binary_cols = ['Churn', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

    for col in binary_cols:
        df_ml[col] = le.fit_transform(df_ml[col])

    df_ml = pd.get_dummies(df_ml)

    print("\n🤖 Dados prontos para Machine Learning!")
    print(df_ml.head())

    return df_ml


# ================================
# 🚀 EXECUÇÃO
# ================================
if __name__ == "__main__":
    df = load_and_clean_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    churn_contract, churn_payment = analyze_churn(df)

    plot_churn_payment(churn_payment)
    plot_churn_contract(churn_contract)

    df_ml = prepare_for_ml(df)

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, confusion_matrix


    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()  # Remove linhas com valores vazios em TotalCharges


    le = LabelEncoder()
    df_model = df.copy()
    for col in df_model.select_dtypes(include=['object']).columns:
        if col != 'customerID':
            df_model[col] = le.fit_transform(df_model[col])


    X = df_model.drop(['customerID', 'Churn'], axis=1)
    y = df_model['Churn']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt


    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()  # Remove valores nulos gerados na conversão

    le = LabelEncoder()
    df_model = df.copy()
    for col in df_model.select_dtypes(include=['object']).columns:
        if col != 'customerID':
            df_model[col] = le.fit_transform(df_model[col])

    X = df_model.drop(['customerID', 'Churn'], axis=1)
    y = df_model['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Relatório de Performance ---")
    print(classification_report(y_test, y_pred))

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Variáveis que mais causam Churn')
    plt.savefig('importancia_variaveis_churn.png', dpi=300, bbox_inches='tight')
    plt.show()


    def predizer_cliente(dados_novo_cliente):

        probabilidade = model.predict_proba(dados_novo_cliente)
        return f"Chance de Churn: {probabilidade[0][1] * 100:.2f}%"


    # Exemplo: Pegando o primeiro cliente do teste para ver a previsão
    print(predizer_cliente(X_test.iloc[[0]]))

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fica', 'Churn'])
    disp.plot(cmap='Blues')
    plt.title('Matriz de Confusão: Acertos vs Erros')
    plt.savefig('matriz_confusao.png', dpi=300)
    plt.show()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import classification_report, roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    print("\n=== Relatório de Classificação ===")
    print(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Desempenho do Modelo (Curva ROC)')
    plt.legend(loc="lower right")

    plt.savefig('02_curva_roc_final.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nSucesso! O gráfico foi salvo e o AUC final é: {auc:.2f}")

    plt.figure(figsize=(10, 6))

    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

    feat_imp.tail(10).plot(kind='barh', color='skyblue')

    plt.title('Top 10 Variáveis que mais influenciam o Churn')
    plt.xlabel('Importância Relativa')
    plt.ylabel('Variáveis')

    plt.savefig('03_feature_importance_final.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n--- Tudo pronto! O gráfico de importância foi salvo. ---")
