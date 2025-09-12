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
# CHARGEMENT DES DONNÉES 💾
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# -------------------------------
# BARRE LATÉRALE - MENU DE NAVIGATION 🧭
# -------------------------------
st.sidebar.title("📌 Menu de navigation")
menu = st.sidebar.radio(
    "",
    ["Accueil", "Aperçu des données", "Statistiques descriptives", "Visualisations", "Corrélations", "Prédiction ML"]
)

# -------------------------------
# PAGE ACCUEIL 🏠
# -------------------------------
if menu == "Accueil":
    st.title("🚢 Titanic Data App")
    st.markdown("""
    Bienvenue dans l’application interactive **Titanic Data Explorer** !
    
    Explorez le dataset du Titanic, **analysez** les données et testez un modèle de Machine Learning pour **prédire** la survie des passagers.
    """)
    try:
        st.image("titanic.png", use_container_width=True)
    except:
        st.warning("⚠️ L'image `titanic.png` est introuvable. Veuillez la placer dans le dossier du projet.")

# -------------------------------
# PAGE 1 : APERÇU DES DONNÉES 📊
# -------------------------------
elif menu == "Aperçu des données":
    st.title("🚢 Aperçu des données")
    st.markdown("---")
    st.subheader("Informations générales")
    st.info(f"**Nombre de lignes :** {df.shape[0]} | **Nombre de colonnes :** {df.shape[1]}")
    st.subheader("Les 20 premières lignes du dataset")
    st.dataframe(df.head(20))

# -------------------------------
# PAGE 2 : STATISTIQUES DESCRIPTIVES 📈
# -------------------------------
elif menu == "Statistiques descriptives":
    st.title("📊 Statistiques descriptives")
    st.markdown("---")
    st.write(df.describe(include="all"))

# -------------------------------
# PAGE 3 : VISUALISATIONS 🖼️
# -------------------------------
elif menu == "Visualisations":
    st.title("📈 Visualisations")
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Répartition des survivants")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Survived", ax=ax, palette="coolwarm")
        plt.title("Répartition des survivants (0 = Décédé, 1 = Survécu)")
        st.pyplot(fig)

    with col2:
        st.subheader("Répartition par sexe")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Sex", ax=ax, palette="viridis")
        plt.title("Répartition des passagers par sexe")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Analyse par variable")
    feature = st.selectbox("Choisissez une colonne :", df.columns)
    fig, ax = plt.subplots()
    if df[feature].dtype == "object":
        sns.countplot(data=df, x=feature, ax=ax, palette="Set2")
        plt.title(f"Distribution de la variable : {feature}")
        plt.xticks(rotation=45)
    else:
        sns.histplot(df[feature], kde=True, ax=ax, color="darkcyan")
        plt.title(f"Distribution de la variable : {feature}")
    st.pyplot(fig)

# -------------------------------
# PAGE 4 : CORRÉLATIONS 🔗
# -------------------------------
elif menu == "Corrélations":
    st.title("🔗 Matrice de corrélation")
    st.markdown("---")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------------
# PAGE 5 : PRÉDICTION ML 🤖
# -------------------------------
elif menu == "Prédiction ML":
    st.title("🤖 Prédiction de survie (Machine Learning)")
    st.markdown("---")

    # Préparation des données
    data = df.copy()
    if "Survived" not in data.columns:
        st.error("❌ La colonne `Survived` est manquante. Le modèle ne peut pas être entraîné.")
    else:
        data = data.dropna(subset=["Age", "Sex", "Pclass"])
        data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

        X = data[["Pclass", "Sex", "Age"]]
        y = data["Survived"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Affichage de la précision
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"**Précision du modèle :** {acc:.2f}")

        # Formulaire de prédiction
        st.subheader("Faites une prédiction pour un passager")
        col_form1, col_form2, col_form3 = st.columns(3)

        with col_form1:
            pclass = st.selectbox("Classe du passager", [1, 2, 3], help="1 = Première classe, 2 = Deuxième, 3 = Troisième")
        with col_form2:
            sex = st.selectbox("Sexe", ["male", "female"])
        with col_form3:
            age = st.slider("Âge", 0, 80, 30)

        input_data = pd.DataFrame({
            "Pclass": [pclass],
            "Sex": [0 if sex == "male" else 1],
            "Age": [age]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction]

        st.markdown("### Résultat de la prédiction")
        if prediction == 1:
            st.success(f"✅ Le modèle prédit que ce passager **aurait survécu** (Probabilité : {probability:.2f})")
        else:
            st.error(f"❌ Le modèle prédit que ce passager **serait décédé** (Probabilité : {probability:.2f})")
