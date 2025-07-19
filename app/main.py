import streamlit as st

st.set_page_config(
    page_title="Decision - Co-Piloto de Recrutamento",
    layout="wide",
    initial_sidebar_state="expanded"
)

pages = {
    "Decision": [
        st.Page("pages/match.py", title="Match perfil"),
        st.Page("pages/sobre.py", title="Sobre")
    ]
}

pg = st.navigation(pages)
pg.run()