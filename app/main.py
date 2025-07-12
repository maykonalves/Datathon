import streamlit as st

st.set_page_config(
    page_title="Decision - Co-Piloto de Recrutamento",
    layout="wide",
    initial_sidebar_state="expanded"
)

pages = {
    "Decision": [
        st.Page("pages/home.py", title="Home"),
        st.Page("pages/match.py", title="Match perfil"),
        st.Page("pages/sobre.py", title="Sobre")
    ],
    "Desenvolvimento": [
        st.Page("pages/eda.py", title="Exploração de Dados"),
        st.Page("pages/treinamento.py", title="Avaliação do Modelo")
    ]
}

pg = st.navigation(pages)
pg.run()

st.markdown(
    """
    Bem-vindo ao Co-Piloto de Recrutamento da Decision!
    
    Esta ferramenta utiliza Inteligência Artificial para otimizar o processo de
    seleção, conectando os melhores talentos às vagas certas.
    
    ### Funcionalidades:
    - **Match Inteligente:** Encontre os candidatos mais compatíveis para uma vaga.
    """
)