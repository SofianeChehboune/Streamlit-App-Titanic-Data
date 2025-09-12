import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# CONFIG PAGE ⚙️
# -------------------------------
st.set_page_config(
    page_title="Titanic App",
    layout="wide",
    page_icon="🚢"
)

# -------------------------------
# CSS CUSTOM 🎨
# -------------------------------
st.markdown("""
<style>
h1, h2, h3 {
    text-align: center;
    color: #1a426e;
}
footer {visibility: hidden;}
.reportview-container {
    background: #f8fbfd;
}
div.stButton > button {
    background-color: #1a426e;
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA 💾
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# -------------------------------
# HEADER 🚢
# -------------------------------
st.title("🚢 Titanic Data Explorer")
st.markdown("### Explorez, analysez et prédisez la survie des passagers du Titanic ⚓")

# -------------------------------
# TABS 📑
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Accueil", "📊 Données", "📈 Visualisations", "🔗 Corrélations", "🤖 Prédiction ML"]
)

# -------------------------------
# TAB 1 : Accueil
# -------------------------------
with tab1:
    st.image("https://i.ibb.co/4YjNQGc/titanic.jpg", use_container_width=True)
    st.success("Bienvenue dans cette app interactive ! Vous pouvez naviguer dans les onglets pour explorer et prédire la survie des passagers.")

# -------------------------------
# TAB 2 : Données
# -------------------------------
with tab2:
    st.subheader("Aperçu du dataset Titanic")
    st.write(df.head(20))

    st.markdown("### Statistiques descriptives")
    st.write(df.describe(include="all"))

# -------------------------------
# TAB 3 : Visualisations
# -------------------------------
with tab3:
    st.subheader("Exploration interactive")
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x="Survived", color="Sex", barmode="group",
                            title="Répartition des survivants par sexe")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x="Pclass", color="Survived", barmode="group",
                            title="Classe des passagers vs Survie")
        st.plotly_chart(fig2, use_container_width=True)

    feature = st.selectbox("Choisissez une variable :", df.columns)
    if df[feature].dtype != "object":
        fig3 = px.histogram(df, x=feature, nbins=30, title=f"Distribution de {feature}", marginal="box")
    else:
        fig3 = px.histogram(df, x=feature, title=f"Distribution de {feature}", color="Survived")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# TAB 4 : Corrélations
# -------------------------------
with tab4:
    st.subheader("Matrice de corrélation")
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="Blues",
        showscale=True
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 5 : Prédiction ML
# -------------------------------
with tab5:
    st.subheader("Prédiction de survie")

    data = df.dropna(subset=["Age", "Sex", "Pclass", "Fare"])
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

    X = data[["Pclass", "Sex", "Age", "Fare"]]
    y = data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Précision du modèle : {acc:.2f}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pclass = st.selectbox("Classe", [1, 2, 3])
    with col2:
        sex = st.selectbox("Sexe", ["male", "female"])
    with col3:
        age = st.slider("Âge", 0, 80, 30)
    with col4:
        fare = st.slider("Tarif du billet", 0, 500, 50)

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [0 if sex == "male" else 1],
        "Age": [age],
        "Fare": [fare]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Ce passager aurait survécu (probabilité : {probability:.2f})")
    else:
        st.error(f"❌ Ce passager n’aurait pas survécu (probabilité : {probability:.2f})")
