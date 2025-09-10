import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# CONFIGURATION DE LA PAGE
# -------------------------------
st.set_page_config(
    page_title="Titanic Data Explorer",
    layout="wide",
    page_icon="🚢"
)

st.title("🚢 Titanic Data Explorer")
st.markdown("Visualisation interactive du dataset Titanic utilisé dans *Titanic_Data.ipynb*")

# -------------------------------
# CHARGEMENT DES DONNÉES
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# -------------------------------
# APERÇU DES DONNÉES
# -------------------------------
st.subheader("Aperçu du Dataset")
st.dataframe(df.head(20))

# Dimensions
st.write(f"**Nombre de lignes :** {df.shape[0]} | **Nombre de colonnes :** {df.shape[1]}")

# -------------------------------
# STATISTIQUES GÉNÉRALES
# -------------------------------
st.subheader("Statistiques descriptives")
st.write(df.describe(include="all"))

# -------------------------------
# VISUALISATIONS
# -------------------------------
st.subheader("Visualisations globales")

col1, col2 = st.columns(2)

with col1:
    st.write("**Répartition des survivants (variable cible si présente)**")
    if "Survived" in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Survived", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("La colonne `Survived` n'existe pas dans ce dataset.")

with col2:
    st.write("**Répartition par sexe**")
    if "Sex" in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Sex", ax=ax, palette="Set2")
        st.pyplot(fig)
    else:
        st.warning("La colonne `Sex` n'existe pas dans ce dataset.")

# -------------------------------
# VISUALISATION DYNAMIQUE
# -------------------------------
st.subheader("Analyse par variable")

feature = st.selectbox("Choisissez une colonne à analyser :", df.columns)

if df[feature].dtype == "object":
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=feature, ax=ax, palette="viridis")
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax, color="blue")
    st.pyplot(fig)

# -------------------------------
# CORRÉLATIONS
# -------------------------------
st.subheader("Matrice de corrélation (variables numériques)")

corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
