import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# Configuration générale
# =========================
st.set_page_config(page_title="🚢 Titanic Data App", layout="wide")

# =========================
# Chargement des données
# =========================
df = pd.read_csv("titanic.csv")

# =========================
# Titre et description
# =========================
st.title("🚢 Titanic Data App")
st.markdown("""
Bienvenue dans l’application interactive **Titanic Data Explorer** !  
Explorez le dataset du Titanic, **analysez** les données et testez un modèle de Machine Learning pour **prédire la survie des passagers**.
""")

# =========================
# Aperçu des données
# =========================
st.header("📊 Aperçu des données")
st.dataframe(df.head())

# =========================
# Nettoyage & Informations
# =========================
st.header("🧹 Informations & Nettoyage")

with st.expander("ℹ️ Infos générales (df.info)"):
    buffer = []
    df.info(buf=buffer.append)
    st.text("".join(buffer))

with st.expander("🔍 Valeurs manquantes (df.isnull)"):
    st.write(df.isnull().sum())

with st.expander("📈 Statistiques descriptives (df.describe)"):
    st.write(df.describe(include="all"))

# =========================
# Visualisations
# =========================
st.header("📉 Visualisations interactives")

# --- Répartition des survivants ---
st.subheader("Répartition des survivants")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="Survived", palette="Set1", ax=ax)
ax.set_title("Répartition des survivants")
st.pyplot(fig, use_container_width=True)

# --- Répartition par sexe ---
st.subheader("Répartition par sexe")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="Sex", palette="Set2", ax=ax)
ax.set_title("Répartition par sexe")
st.pyplot(fig, use_container_width=True)

# --- Analyse personnalisée ---
st.subheader("Analyse personnalisée")
feature = st.selectbox("Choisissez une colonne :", df.columns)

fig, ax = plt.subplots(figsize=(6, 4))
if df[feature].dtype == "object":
    sns.countplot(data=df, x=feature, ax=ax, palette="Set3")
    ax.set_title(f"Distribution de la variable : {feature}")
    plt.xticks(rotation=45)
else:
    sns.histplot(df[feature], kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution de la variable : {feature}")

st.pyplot(fig, use_container_width=True)

# =========================
# Modélisation ML
# =========================
st.header("🤖 Prédiction de survie")

# Préparation des données
df_ml = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]].copy()
df_ml["Sex"] = df_ml["Sex"].map({"male": 0, "female": 1})
df_ml = df_ml.dropna()

X = df_ml.drop("Survived", axis=1)
y = df_ml["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"🎯 **Précision du modèle : {acc:.2f}**")

# =========================
# Formulaire de prédiction
# =========================
st.subheader("Faites votre prédiction")

pclass = st.selectbox("Classe", [1, 2, 3])
sex = st.radio("Sexe", ["male", "female"])
age = st.slider("Âge", 0, 80, 25)
sibsp = st.number_input("Nombre de frères/soeurs / conjoints à bord", 0, 10, 0)
parch = st.number_input("Nombre de parents / enfants à bord", 0, 10, 0)
fare = st.number_input("Tarif du billet", 0.0, 600.0, 32.2)

# Conversion
sex_val = 0 if sex == "male" else 1
input_data = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare]],
                          columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

st.markdown("### Résultat")
if prediction == 1:
    st.success(f"✅ Ce passager aurait survécu (probabilité : {probability:.2f})")

    # 🎭 Cas spécial Jack Dawson
    if pclass == 3 and sex == "male" and age == 19 and probability > 0.90:
        st.info("🎭 Cela ressemble étrangement à **Jack Dawson**... 🖼️")
else:
    st.error(f"❌ Ce passager n’aurait pas survécu (probabilité : {probability:.2f})")

    if pclass == 3 and sex == "male" and age == 19:
        st.warning("💔 Comme dans l’histoire, **Jack Dawson** n’a pas eu de place sur le radeau...")
