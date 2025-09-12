import io
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------- CONFIG ----------
st.set_page_config(page_title="Titanic App", layout="wide", page_icon="🚢")
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set_theme(style="whitegrid")

# ---------- CSS (garde ton style) ----------
st.markdown("""
<style>
/* Fond clair global */
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
    background: #0d6efd;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    padding: 0.6em 1.2em;
    border: none;
    transition: 0.3s;
}
div.stButton > button:hover {
    background: #084298;
    transform: scale(1.05);
}
section[data-testid="stSidebar"] {
    background: #f4f6f9;
    color: black;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df_orig = load_data()

# copie de travail dans session_state (pour nettoyages interactifs sans toucher au CSV)
if 'df_work' not in st.session_state:
    st.session_state.df_work = df_orig.copy()

df = st.session_state.df_work  # on utilise la copie de travail partout

# ---------- SIDEBAR ----------
st.sidebar.title("📌 Menu de navigation")
menu = st.sidebar.radio("", [
    "🏠 Accueil",
    "📊 Aperçu des données",
    "🧹 Nettoyage & Préparation",
    "📈 Statistiques descriptives",
    "📉 Visualisations",
    "🔗 Corrélations",
    "🤖 Prédiction ML"
])

# ---------- PAGE ACCUEIL ----------
if menu == "🏠 Accueil":
    st.markdown(
        """
        <div style="background: linear-gradient(160deg, #e6ecf5, #a3b6d9 40%, #2c3e50 100%);
                    border-radius: 15px; padding: 30px; text-align: center; box-shadow: 0px 4px 25px rgba(0,0,0,0.3);">
            <h1 style="font-size: 2.6em; color:#0d1b2a;">🚢 Titanic Data App</h1>
            <p style="font-size: 1.2em; color:#1a1a1a;">
            Bienvenue dans l’application interactive <b>Titanic Data Explorer</b> !<br>
            Explorez le dataset du Titanic, <b>analysez</b> les données et testez un modèle de 
            Machine Learning pour <b>prédire la survie des passagers</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    try:
        st.image("titanic.png", use_container_width=True, caption="Légendaire Titanic ⚓")
    except:
        st.warning("⚠️ L'image `titanic.png` est introuvable. (vérifie le nom exact)")

# ---------- PAGE: APERÇU ----------
elif menu == "📊 Aperçu des données":
    st.title("📊 Aperçu des données")
    st.markdown("---")
    st.info(f"**Nombre de lignes :** {df.shape[0]} | **Nombre de colonnes :** {df.shape[1]}")
    st.dataframe(df.head(20), use_container_width=True)

# ---------- PAGE: NETTOYAGE ----------
elif menu == "🧹 Nettoyage & Préparation":
    st.title("🧹 Nettoyage & Préparation des données")
    st.markdown("---")

    # df.info() propre
    st.subheader("📌 Informations générales (df.info)")
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    st.text(info_str)

    # valeurs manquantes
    st.subheader("📌 Valeurs manquantes")
    na_counts = df.isnull().sum().sort_values(ascending=False)
    st.write(na_counts[na_counts > 0] if not na_counts[na_counts > 0].empty else "Aucune valeur manquante détectée.")

    # doublons
    st.subheader("📌 Doublons")
    dup_count = df.duplicated().sum()
    st.write(f"Nombre de doublons : **{dup_count}**")
    if dup_count > 0:
        if st.button("Supprimer les doublons"):
            st.session_state.df_work = st.session_state.df_work.drop_duplicates().reset_index(drop=True)
            df = st.session_state.df_work
            st.success("Doublons supprimés.")

    # remplir Age par médiane
    if "Age" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Remplir les NaN d'Age par la médiane"):
                med = st.session_state.df_work["Age"].median()
                st.session_state.df_work["Age"] = st.session_state.df_work["Age"].fillna(med)
                st.success(f"NaN d'Age remplacés par la médiane = {med:.1f}")
        with col2:
            if st.button("Supprimer les lignes avec NaN critiques (Age/Pclass/Sex)"):
                before = len(st.session_state.df_work)
                st.session_state.df_work = st.session_state.df_work.dropna(subset=["Age", "Pclass", "Sex"]).reset_index(drop=True)
                after = len(st.session_state.df_work)
                st.success(f"Lignes supprimées: {before - after}")

    # supprimer colonnes
    st.subheader("📌 Supprimer des colonnes (optionnel)")
    cols_to_drop = st.multiselect("Choisis les colonnes à supprimer (local, pas le CSV)", df.columns.tolist())
    if cols_to_drop:
        if st.button("Appliquer suppression colonnes"):
            st.session_state.df_work = st.session_state.df_work.drop(columns=cols_to_drop)
            df = st.session_state.df_work
            st.success(f"Colonnes supprimées : {cols_to_drop}")

    st.markdown("---")
    st.subheader("Aperçu de la copie de travail (5 premières lignes)")
    st.dataframe(st.session_state.df_work.head(), use_container_width=True)

# ---------- PAGE: STAT DESCRIPTIVES ----------
elif menu == "📈 Statistiques descriptives":
    st.title("📈 Statistiques descriptives")
    st.markdown("---")
    st.write(df.describe(include="all"))

# ---------- PAGE: VISUALISATIONS ----------
elif menu == "📉 Visualisations":
    st.title("📉 Visualisations interactives")
    st.markdown("---")

    col1, col2 = st.columns(2)

    # Répartition des survivants
    with col1:
        st.subheader("Répartition des survivants")
        if "Survived" in df.columns:
            counts = df["Survived"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="Blues")
            ax.set_xlabel("Survived")
            ax.set_ylabel("Nombre")
            ax.set_title("Répartition des survivants (0 = Décédé, 1 = Survécu)")
            st.pyplot(fig, use_container_width=True)
        else:
            st.warning("La colonne 'Survived' est introuvable dans le dataset.")

    # Répartition par sexe
    with col2:
        st.subheader("Répartition par sexe")
        if "Sex" in df.columns:
            counts = df["Sex"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Pastel1")
            ax.set_xlabel("Sexe")
            ax.set_ylabel("Nombre")
            ax.set_title("Répartition par sexe")
            st.pyplot(fig, use_container_width=True)
        else:
            st.warning("La colonne 'Sex' est introuvable dans le dataset.")

    st.markdown("---")
    st.subheader("Analyse personnalisée")
    feature = st.selectbox("Choisissez une colonne :", df.columns, index=0)

    # ---- logique d'affichage adaptée ----
    # Si categorical (object) ou peu de valeurs uniques -> barplot horizontal
    # Si numeric -> histogramme
    max_cat_display = st.slider("Top N pour les catégories (si applicable)", min_value=5, max_value=30, value=10)

    if feature:
        unique_count = df[feature].nunique(dropna=True)
        if df[feature].dtype == "object" or unique_count <= max_cat_display:
            # categories: on affiche top N pour éviter fouillis
            top_n = max(3, min(max_cat_display, unique_count))
            counts = df[feature].value_counts().nlargest(top_n)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="Set2")
            ax.set_xlabel("Nombre")
            ax.set_ylabel(feature)
            ax.set_title(f"Top {top_n} catégories pour : {feature}")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            # numeric large-unique -> histogramme
            try:
                series = pd.to_numeric(df[feature], errors='coerce').dropna()
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(series, kde=True, ax=ax)
                ax.set_title(f"Histogramme de : {feature}")
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Impossible de tracer {feature} : {e}")

# ---------- PAGE: CORRÉLATIONS ----------
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

# ---------- PAGE: PREDICTION ----------
elif menu == "🤖 Prédiction ML":
    st.title("🤖 Prédiction de survie (Machine Learning)")
    st.markdown("---")

    data = st.session_state.df_work.copy()  # on entraîne sur la copie de travail
    if "Survived" not in data.columns:
        st.error("❌ La colonne `Survived` est manquante.")
    else:
        # simple préparation (garder Pclass, Sex, Age)
        if not set(["Pclass", "Sex", "Age"]).issubset(data.columns):
            st.error("Les colonnes requises (Pclass, Sex, Age) ne sont pas toutes présentes.")
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
                # clin d'oeil Jack (optionnel)
                if pclass == 3 and sex == "male" and age == 19 and probability > 0.9:
                    st.info("🎭 Ça ressemble à Jack Dawson...")
            else:
                st.error(f"❌ Ce passager n’aurait pas survécu (probabilité : {probability:.2f})")
                if pclass == 3 and sex == "male" and age == 19 and probability > 0.9:
                    st.warning("💔 Comme dans le film...")

