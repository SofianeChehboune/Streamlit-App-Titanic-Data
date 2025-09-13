import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# CONFIGURATION DE LA PAGE ⚙️
# -------------------------------
st.set_page_config(
    page_title="Titanic - Le Destin des Passagers",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# STYLE CSS PERSONNALISÉ 🎨
# -------------------------------
st.markdown("""
<style>
    /* Thème général */
    body {
        color: #333;
        background-color: #f0f2f6;
    }
    .stApp {
        background-image: url("https://www.transparenttextures.com/patterns/light-paper-fibers.png");
        background-attachment: fixed;
    }
    h1, h2, h3 {
        font-family: 'Garamond', serif;
        color: #0a2a43;
        text-shadow: 2px 2px 4px #ccc;
    }
    .stButton>button {
        background-color: #0a2a43;
        color: white;
        border-radius: 50px;
        transition: all 0.3s ease-in-out;
        border: 2px solid #0a2a43;
    }
    .stButton>button:hover {
        background-color: white;
        color: #0a2a43;
        transform: scale(1.05);
        border: 2px solid #0a2a43;
    }
    .stMetric {
        background-color: #ffffff;
        border-left: 5px solid #0a2a43;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# CHARGEMENT DES DONNÉES 💾
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("titanic.csv")
    # Nettoyage de base
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(columns=['Cabin'], inplace=True)
    return df

df = load_data()

# -------------------------------
# BARRE LATÉRALE - MENU 🧭
# -------------------------------
st.sidebar.image("titanic.jpg", use_container_width=True)
st.sidebar.title("🚢 Menu de Navigation")

pages = {
    "🏠 Accueil": "house",
    "📊 Tableau de Bord": "bar-chart-fill",
    "📈 Exploration Détaillée": "search",
    "🤖 Modèle de Prédiction": "robot"
}

selection = st.sidebar.radio(" ", list(pages.keys()))

# -------------------------------
# FONCTIONS DES PAGES
# -------------------------------

def page_accueil():
    st.title("🚢 Le Destin des Passagers du Titanic")
    st.markdown("### Une analyse interactive et prédictive de la célèbre tragédie.")
    
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.8); padding: 2rem; border-radius: 15px; box-shadow: 0 8px 16px rgba(0,0,0,0.1);">
    Bienvenue dans cette application qui vous plonge au cœur des données du Titanic. Explorez les statistiques, visualisez les répartitions des passagers et découvrez les facteurs qui ont influencé leur survie.
    <br><br>
    Utilisez le menu à gauche pour naviguer entre les différentes sections :
    <ul>
        <li><b>Tableau de Bord :</b> Un aperçu global des chiffres clés.</li>
        <li><b>Exploration Détaillée :</b> Des graphiques interactifs pour creuser les données.</li>
        <li><b>Modèle de Prédiction :</b> Testez un modèle de Machine Learning pour prédire la survie d'un passager.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.image("titanic .png", use_container_width=True, caption="Le Titanic, un navire de légende.")

def page_dashboard():
    st.title("📊 Tableau de Bord")
    st.markdown("---")

    # KPIs
    total_passengers = len(df)
    survivors = df['Survived'].sum()
    survival_rate = survivors / total_passengers * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Passagers", f"{total_passengers} 👥")
    col2.metric("Survivants", f"{survivors} 😊")
    col3.metric("Taux de Survie", f"{survival_rate:.2f}% 💔")

    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        # Sunburst Chart pour la répartition des survivants
        st.subheader("Répartition des Survivants")
        fig = px.sunburst(df, path=['Pclass', 'Sex', 'Survived'], 
                          values='PassengerId', 
                          color='Survived',
                          color_discrete_map={1:'green', 0:'red'},
                          title="Survivants par Classe et Sexe")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Histogramme de l'âge
        st.subheader("Distribution de l'Âge")
        fig = px.histogram(df, x='Age', nbins=40, color='Survived',
                           marginal="box", # or violin, rug
                           title="Distribution de l'Âge des Passagers")
        st.plotly_chart(fig, use_container_width=True)

def page_exploration():
    st.title("📈 Exploration Détaillée")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Options de Visualisation")
        x_axis = st.selectbox("Choisissez l'axe X", df.columns, index=list(df.columns).index('Pclass'))
        y_axis = st.selectbox("Choisissez l'axe Y", df.columns, index=list(df.columns).index('Age'))
        color_axis = st.selectbox("Choisissez la couleur", df.columns, index=list(df.columns).index('Survived'))

    with col2:
        st.subheader(f"Analyse de {x_axis} vs {y_axis}")
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_axis,
                         hover_name="Name", size='Fare',
                         title=f"{x_axis} vs {y_axis} coloré par {color_axis}")
        st.plotly_chart(fig, use_container_width=True)

    # Matrice de corrélation
    st.subheader("Matrice de Corrélation")
    corr = df.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(
                   z=corr.values,
                   x=corr.columns,
                   y=corr.columns,
                   colorscale='Blues',
                   colorbar=dict(title='Corrélation')))
    fig.update_layout(title='Corrélation entre les variables numériques')
    st.plotly_chart(fig, use_container_width=True)


def page_prediction():
    st.title("🤖 Modèle de Prédiction de Survie")
    st.markdown("---")

    # Préparation des données et entraînement du modèle
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    df_ml = df.dropna(subset=features)
    df_ml['Sex'] = df_ml['Sex'].map({'male': 0, 'female': 1})
    
    X = df_ml[features]
    y = df_ml['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.info(f"**Précision du modèle sur les données de test :** {accuracy:.2%}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔮 Faites votre prédiction")
        pclass = st.selectbox("Classe du passager", [1, 2, 3])
        sex = st.selectbox("Sexe", ["male", "female"])
        age = st.slider("Âge", 0, 80, 25)
        sibsp = st.slider("Nombre de frères/sœurs/époux(ses)", 0, 8, 0)
        parch = st.slider("Nombre de parents/enfants", 0, 6, 0)
        fare = st.slider("Tarif du billet", 0, 512, 32)

        input_data = pd.DataFrame({
            'Pclass': [pclass], 'Sex': [0 if sex == 'male' else 1], 'Age': [age],
            'SibSp': [sibsp], 'Parch': [parch], 'Fare': [fare]
        })

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

    with col2:
        st.subheader("📊 Résultat de la Prédiction")
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = proba[1] * 100,
            title = {'text': "Probabilité de Survie"},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': "green" if prediction == 1 else "red"},
                     'steps' : [
                         {'range': [0, 50], 'color': "lightgray"},
                         {'range': [50, 100], 'color': "gray"}]
                    },
            domain = {'x': [0, 1], 'y': [0, 1]}
        ))
        st.plotly_chart(fig, use_container_width=True)

        if prediction == 1:
            st.success(f"🎉 **Le passager aurait probablement survécu !** (Probabilité: {proba[1]:.2%})")
        else:
            st.error(f"💔 **Le passager n'aurait probablement pas survécu.** (Probabilité: {proba[0]:.2%})")

    # Feature Importance
    st.subheader("Qu'est-ce qui influence le plus la survie ?")
    feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    fig = px.bar(feature_importance.sort_values('importance', ascending=False), 
                 x='importance', y='feature', orientation='h',
                 title="Importance des Caractéristiques dans le Modèle")
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# ROUTEUR PRINCIPAL
# -------------------------------
if selection == "🏠 Accueil":
    page_accueil()
elif selection == "📊 Tableau de Bord":
    page_dashboard()
elif selection == "📈 Exploration Détaillée":
    page_exploration()
elif selection == "🤖 Modèle de Prédiction":
    page_prediction()
