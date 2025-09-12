import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# CONFIGURATION DE LA PAGE ⚙️
# -------------------------------
st.set_page_config(
    page_title="Titanic App",
    layout="wide",
    page_icon="🚢"
)

# -------------------------------
# STYLE CSS PERSONNALISÉ 🎨
# -------------------------------
st.markdown("""
<style>
/* Fond général clair */
body {
    background-color: #ffffff;
    color: #000000;
}
h1, h2, h3 {
    color: #0d3b66;
    text-align: center;
    font-weight: bold;
}
div.stButton > button {
    background: #0d6efd;  /* Bleu vif */
    color: white;
    border-radius: 10px;
    font-weight: bold;
    padding: 0.6em 1.2em;
    border: none;
    transition: 0.3s;
}
div.stButton > button:hover {
    background: #084298;  /* Bleu foncé au hover */
    transform: scale(1.05);
}
/* Sidebar en gris clair */
section[data-testid="stSidebar"] {
    background: #f4f6f9;
    color: black;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# CHARGEMENT DES DONNÉES 💾
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# -------------------------------
# BARRE LATÉRALE - MENU 🧭
# -------------------------------
st.sidebar.title("📌 Menu de navigation")
menu = st.sidebar.radio(
    "",
    ["🏠 Accueil", "📊 Aperçu des données", "📈 Statistiques descriptives", "📉 Visualisations", "🔗 Corrélations", "🤖 Prédiction ML"]
)

# -------------------------------
# PAGE ACCUEIL 🏠
# -------------------------------
if menu == "🏠 Accueil":
    st.title("🚢 Titanic Data App")
    st.markdown("""
    Bienvenue dans l’application interactive **Titanic Data Explorer** !  
    Explorez le dataset du Titanic, **analysez** les données et testez un modèle de Machine Learning pour **prédire la survie des passagers**.
    """)
    try:
        st.image("titanic app.png", use_container_width=True, caption="Légendaire Titanic ⚓")
    except:
        st.warning("⚠️ L'image `titanic.png` est introuvable.")

# -------------------------------
# PAGE 1 : APERÇU DES DONNÉES 📊
# -------------------------------
elif menu == "📊 Aperçu des données":
    st.title("📊 Aperçu des données")
    st.markdown("---")
    st.info(f"**Nombre de lignes :** {df.shape[0]} | **Nombre de colonnes :** {df.shape[1]}")
    st.dataframe(df.head(20), use_container_width=True)

# -------------------------------
# PAGE 2 : STATISTIQUES DESCRIPTIVES 📈
# -------------------------------
elif menu == "📈 Statistiques descriptives":
    st.title("📈 Statistiques descriptives")
    st.markdown("---")
    st.write(df.describe(include="all"))

# -------------------------------
# PAGE 3 : VISUALISATIONS 🖼️
# -------------------------------
elif menu == "📉 Visualisations":
    st.title("📉 Visualisations interactives")
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Répartition des survivants")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Survived", ax=ax, palette="Blues")
        ax.set_title("Répartition des survivants (0 = Décédé, 1 = Survécu)")
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Répartition par sexe")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Sex", ax=ax, palette="Pastel1")
        ax.set_title("Répartition par sexe")
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Analyse personnalisée")
    feature = st.selectbox("Choisissez une colonne :", df.columns)
    fig, ax = plt.subplots()
    if df[feature].dtype == "object":
        sns.countplot(data=df, x=feature, ax=ax, palette="Set2")
        ax.set_title(f"Distribution de la variable : {feature}")
        plt.xticks(rotation=45)
    else:
        sns.histplot(df[feature], kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution de la variable : {feature}")
    st.pyplot(fig, use_container_width=True)

# -------------------------------
# PAGE 4 : CORRÉLATIONS 🔗
# -------------------------------
elif menu == "🔗 Corrélations":
    st.title("🔗 Matrice de corrélation")
    st.markdown("---")
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        st.warning("⚠️ Pas de colonnes numériques à corréler.")
    else:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, annot_kws={"fontsize": 10})
        st.pyplot(fig, use_container_width=True)

# -------------------------------
# PAGE 5 : PRÉDICTION ML 🤖
# -------------------------------
elif menu == "🤖 Prédiction ML":
    st.title("🤖 Prédiction de survie (Machine Learning)")
    st.markdown("---")

    data = df.copy()
    if "Survived" not in data.columns:
        st.error("❌ La colonne `Survived` est manquante.")
    else:
        data = data.dropna(subset=["Age", "Sex", "Pclass"])
        data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

        X = data[["Pclass", "Sex", "Age"]]
        y = data["Survived"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"**Précision du modèle :** {acc:.2f}")

        st.subheader("Faites une prédiction")
        col1, col2, col3 = st.columns(3)
        with col1:
            pclass = st.selectbox("Classe", [1, 2, 3])
        with col2:
            sex = st.selectbox("Sexe", ["male", "female"])
        with col3:
            age = st.slider("Âge", 0, 80, 30)

        input_data = pd.DataFrame({
            "Pclass": [pclass],
            "Sex": [0 if sex == "male" else 1],
            "Age": [age]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction]

        st.markdown("### Résultat")
        if prediction == 1:
            st.success(f"✅ Ce passager aurait survécu (probabilité : {probability:.2f})")
        else:
            st.error(f"❌ Ce passager n’aurait pas survécu (probabilité : {probability:.2f})")
