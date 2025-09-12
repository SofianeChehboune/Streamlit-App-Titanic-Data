import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# CONFIG PAGE
# -------------------------------
st.set_page_config(
    page_title="Titanic App",
    layout="wide",
    page_icon="🚢"
)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# -------------------------------
# SIDEBAR MENU
# -------------------------------
menu = st.sidebar.radio(
    "📌 # Navigation",
    ["## Accueil", "## Aperçu des données", " ## Statistiques descriptives", "## Visualisations", "## Corrélations", "## Prédiction ML"]
)

# -------------------------------
# PAGE ACCUEIL
# -------------------------------
if menu == "Accueil":
    st.title("🚢 Titanic Data App")
    st.markdown("""
    Bienvenue dans l’application interactive **Titanic Data Explorer**.  
    Explorez le dataset du Titanic, analysez les données et testez un modèle de Machine Learning pour prédire la survie des passagers.
    """)
    
    try:
        st.image("titanic.png", use_container_width=True)
    except:
        st.warning("⚠️ Image Titanic non trouvée. Placez un fichier `titanic.jpg` dans le dossier du projet.")

# -------------------------------
# PAGE 1 : OVERVIEW
# -------------------------------
elif menu == "Aperçu des données":
    st.title("🚢 Titanic - Aperçu des données")
    st.write(f"**Nombre de lignes :** {df.shape[0]} | **Nombre de colonnes :** {df.shape[1]}")
    st.dataframe(df.head(20))

# -------------------------------
# PAGE 2 : STATISTICS
# -------------------------------
elif menu == "Statistiques descriptives":
    st.title("📊 Statistiques descriptives")
    st.write(df.describe(include="all"))

# -------------------------------
# PAGE 3 : VISUALISATIONS
# -------------------------------
elif menu == "Visualisations":
    st.title("📈 Visualisations")
    
    col1, col2 = st.columns(2)

    with col1:
        if "Survived" in df.columns:
            st.subheader("Répartition des survivants")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x="Survived", ax=ax)
            st.pyplot(fig)

    with col2:
        if "Sex" in df.columns:
            st.subheader("Répartition par sexe")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x="Sex", palette="Set2", ax=ax)
            st.pyplot(fig)

    st.subheader("Analyse par variable")
    feature = st.selectbox("Choisissez une colonne :", df.columns)
    if df[feature].dtype == "object":
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=feature, palette="viridis", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, color="blue", ax=ax)
        st.pyplot(fig)

# -------------------------------
# PAGE 4 : CORRELATIONS
# -------------------------------
elif menu == "Corrélations":
    st.title("🔗 Matrice de corrélation")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------------
# PAGE 5 : ML PREDICTION
# -------------------------------
elif menu == "Prédiction ML":
    st.title("🤖 Prédiction de survie (ML)")

    # Préparation des données
    data = df.copy()
    if "Survived" not in data.columns:
        st.error("La colonne `Survived` est manquante dans le dataset.")
    else:
        data = data.dropna(subset=["Age", "Sex", "Pclass"])  # nettoyage minimal
        data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

        X = data[["Pclass", "Sex", "Age"]]
        y = data["Survived"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Accuracy
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Précision du modèle :** {acc:.2f}")

        # Formulaire utilisateur
        st.subheader("Faites une prédiction")
        pclass = st.selectbox("Classe du passager (1=1ère, 2=2ème, 3=3ème)", [1, 2, 3])
        sex = st.selectbox("Sexe", ["male", "female"])
        age = st.slider("Âge", 0, 80, 30)

        input_data = pd.DataFrame({
            "Pclass": [pclass],
            "Sex": [0 if sex == "male" else 1],
            "Age": [age]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction]

        if prediction == 1:
            st.success(f"✅ Le modèle prédit : Survie (probabilité {probability:.2f})")
        else:
            st.error(f"❌ Le modèle prédit : Décès (probabilité {probability:.2f})")
