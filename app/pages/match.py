import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import unicodedata
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from unidecode import unidecode

# ==============================================================================
# 1. CONFIGURAÇÃO E CARREGAMENTO DOS RECURSOS
# ==============================================================================

def inicializar_recursos_linguagem():
    """
    Verifica e baixa os recursos necessários do NLTK de forma robusta.
    """
    try:
        stopwords.words('portuguese')
    except (LookupError, OSError):
        st.info("Baixando recursos de linguagem (stopwords) pela primeira vez...")
        nltk.download('stopwords', quiet=True)

inicializar_recursos_linguagem()

@st.cache_resource
def carregar_recursos():
    """
    Carrega o pipeline treinado (.pkl), o modelo SBERT e o limiar de decisão.
    """
    model_dir = 'models'
    pipeline_path = os.path.join(model_dir, 'xgb_pipeline_treinado.pkl')
    threshold_path = os.path.join(model_dir, 'limiar_xgb.txt')

    if not os.path.exists(pipeline_path) or not os.path.exists(threshold_path):
        st.error(
            f"Arquivo do modelo ('{pipeline_path}') ou do limiar não encontrado. "
            "Por favor, execute o script de treinamento `python models/modelo.py` primeiro."
        )
        st.stop()

    pipeline = joblib.load(pipeline_path)
    with open(threshold_path, 'r') as f:
        threshold = float(f.read())
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    return pipeline, sbert_model, threshold

pipeline, sbert_model, threshold = carregar_recursos()

# ==============================================================================
# 2. FUNÇÕES DE PRÉ-PROCESSAMENTO E ENGENHARIA DE FEATURES
# ==============================================================================

MAPA_IDIOMAS = {
    'Nenhum': 0, 'Básico': 1, 'Técnico': 2, 'Intermediário': 3,
    'Avançado': 4, 'Fluente': 5
}
HIERARQUIA_CARGOS = {
    'aprendiz': 1, 'trainee': 2, 'auxiliar': 3, 'assistente': 4,
    'analista': 5, 'junior': 6, 'pleno': 7, 'senior': 8,
    'especialista': 9, 'supervisor': 10, 'lider': 11,
    'coordenador': 12, 'gerente': 13
}
STOP_WORDS = set(stopwords.words('portuguese'))

def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return ' '.join(tokens)

def get_seniority_level(text):
    palavras_chave = HIERARQUIA_CARGOS.keys()
    niveis_encontrados = [p for p in palavras_chave if p in text.split()]
    if not niveis_encontrados: return None
    return max(niveis_encontrados, key=lambda x: HIERARQUIA_CARGOS[x])

def criar_features(df):
    df_copy = df.copy()
    
    df_copy['texto_candidato'] = (
        df_copy['cv'].fillna('') + ' ' + df_copy['conhecimentos_tecnicos_candidato'].fillna('') + ' ' +
        df_copy['nivel_profissional_candidato'].fillna('') + ' ' + df_copy['titulo_candidato'].fillna('') + ' ' +
        df_copy['objetivo_candidato'].fillna('')
    )
    df_copy['texto_vaga'] = (
        df_copy['titulo_vaga'].fillna('') + ' ' + df_copy['area_atuacao_vaga'].fillna('') + ' ' +
        df_copy['principais_atividades_vaga'].fillna('') + ' ' + df_copy['competencias_tecnicas_vaga'].fillna('')
    )
    df_copy['texto_vaga_tratado'] = df_copy['texto_vaga'].apply(preprocess_text)
    df_copy['texto_candidato_tratado'] = df_copy['texto_candidato'].apply(preprocess_text)
    
    for col in ['nivel_ingles_vaga', 'nivel_espanhol_vaga', 'nivel_ingles_candidato', 'nivel_espanhol_candidato']:
        df_copy[col] = df_copy[col].replace('', 'Nenhum').fillna('Nenhum').str.strip().str.title()
        df_copy[col + '_num'] = df_copy[col].map(MAPA_IDIOMAS)
    
    df_copy['match_nivel_ingles'] = (df_copy['nivel_ingles_candidato_num'] >= df_copy['nivel_ingles_vaga_num']).astype(int)
    df_copy['match_nivel_espanhol'] = (df_copy['nivel_espanhol_candidato_num'] >= df_copy['nivel_espanhol_vaga_num']).astype(int)

    df_copy['local_vaga_tratado'] = df_copy['local_vaga'].apply(lambda x: unidecode(str(x)).lower().strip() if pd.notna(x) else '')
    df_copy['local_candidato_tratado'] = df_copy['local_candidato'].apply(lambda x: unidecode(str(x).split(',')[0]).lower().strip() if pd.notna(x) else '')
    df_copy['match_local_cidade_bin'] = (df_copy['local_vaga_tratado'] == df_copy['local_candidato_tratado']).astype(int)
    
    df_copy['nivel_profissional_vaga'] = df_copy['nivel_profissional_vaga'].apply(lambda x: unidecode(str(x)).lower().strip() if pd.notna(x) else None)
    df_copy['maior_nivel_detectado'] = df_copy['texto_candidato_tratado'].apply(get_seniority_level)
    
    def comparar_niveis(row):
        detectado, real = row['maior_nivel_detectado'], row['nivel_profissional_vaga']
        if pd.isna(detectado) or pd.isna(real) or real not in HIERARQUIA_CARGOS: return 0
        return 1 if HIERARQUIA_CARGOS.get(detectado, 0) >= HIERARQUIA_CARGOS.get(real, 0) else 0

    df_copy['match_nivel_bin'] = df_copy.apply(comparar_niveis, axis=1)

    def calcular_diferenca(row):
        detectado, real = row['maior_nivel_detectado'], row['nivel_profissional_vaga']
        if pd.isna(detectado) or pd.isna(real) or real not in HIERARQUIA_CARGOS: return 0
        return HIERARQUIA_CARGOS.get(detectado, 0) - HIERARQUIA_CARGOS.get(real, 0)

    df_copy['diferenca_nivel_senioridade'] = df_copy.apply(calcular_diferenca, axis=1)

    return df_copy

def criar_vetor_final(df_processado, sbert_model_loaded):
    X_vaga = sbert_model_loaded.encode(df_processado['texto_vaga_tratado'].tolist(), show_progress_bar=False)
    X_candidato = sbert_model_loaded.encode(df_processado['texto_candidato_tratado'].tolist(), show_progress_bar=False)
    
    features_adicionais = df_processado[[
        'match_nivel_bin', 'match_nivel_ingles', 'match_nivel_espanhol',
        'diferenca_nivel_senioridade', 'match_local_cidade_bin'
    ]].values
    
    return np.hstack((X_vaga, X_candidato, features_adicionais))

# ==============================================================================
# 3. INTERFACE DO STREAMLIT
# ==============================================================================

st.set_page_config(layout="wide", page_title="Análise de Match")
st.title("Análise de Match: Vaga vs. Candidato")
st.markdown("Insira as informações da vaga e do candidato para calcular a probabilidade de sucesso.")

col1, col2 = st.columns(2)

# --- COLUNA DA VAGA ---
with col1:
    st.subheader("Informações da Vaga")
    titulo_vaga = st.text_input("Título da Vaga", "Desenvolvedor Python Sênior")
    principais_atividades_vaga = st.text_area("Principais Atividades", "Desenvolvimento de APIs REST com Django e Flask.", height=125)
    competencias_tecnicas_vaga = st.text_area("Competências", "Experiência com AWS e Docker. Conhecimento em SQL e bancos de dados PostgreSQL.", height=125)
    nivel_profissional_vaga = st.selectbox("Nível Profissional (Vaga)", list(HIERARQUIA_CARGOS.keys()), index=7) # senior
    nivel_ingles_vaga = st.selectbox("Nível de Inglês (Vaga)", list(MAPA_IDIOMAS.keys()), index=4) # Avançado
    local_vaga = st.text_input("Local da Vaga (Cidade)", "São Paulo")

# --- COLUNA DO CANDIDATO ---
with col2:
    st.subheader("Informações do Candidato")
    titulo_candidato = st.text_input("Título Profissional do Candidato", "Engenheiro de Software")
    objetivo_candidato = st.text_input("Objetivo Profissional", "Atuar como Desenvolvedor Python Sênior")
    cv_pt = st.text_area("Currículo / Resumo do Candidato", "Vasta experiência como desenvolvedor Python, criando soluções escaláveis na nuvem (AWS). Domínio de PostgreSQL e desenvolvimento de microserviços com Docker e Kubernetes.", height=125)
    
    # <<< CORREÇÃO: Aumentada a altura do campo de texto de 50 para 125 pixels >>>
    conhecimentos_tecnicos_candidato = st.text_area("Conhecimentos Técnicos Adicionais", "Microsserviços, Testes automatizados, CI/CD", height=125)
    
    nivel_profissional_candidato = st.selectbox("Nível Profissional (Candidato)", list(HIERARQUIA_CARGOS.keys()), index=7) # senior
    nivel_ingles_candidato = st.selectbox("Nível de Inglês (Candidato)", list(MAPA_IDIOMAS.keys()), index=5) # Fluente
    local_candidato = st.text_input("Local do Candidato (Cidade, Estado)", "São Paulo, SP")

# --- BARRA LATERAL E BOTÃO DE AÇÃO ---
st.sidebar.title("Configurações")
limiar_escolhido = st.sidebar.slider("Limiar de Decisão para 'Bom Match'", 0.0, 1.0, threshold, 0.01)
st.sidebar.info(f"O limiar otimizado do modelo é {threshold*100:.0f}%. Scores acima do limiar escolhido serão considerados um bom match.")

if st.button("Calcular Match", type="primary", use_container_width=True):
    with st.spinner('Analisando perfil e calculando score...'):
        dados_input = pd.DataFrame([{
            'titulo_vaga': titulo_vaga, 'principais_atividades_vaga': principais_atividades_vaga,
            'competencias_tecnicas_vaga': competencias_tecnicas_vaga, 'nivel_profissional_vaga': nivel_profissional_vaga,
            'nivel_ingles_vaga': nivel_ingles_vaga, 'local_vaga': local_vaga,
            'nivel_espanhol_vaga': 'Nenhum', 'area_atuacao_vaga': '', 
            'objetivo_candidato': objetivo_candidato, 'cv': cv_pt,
            'nivel_profissional_candidato': nivel_profissional_candidato, 'nivel_ingles_candidato': nivel_ingles_candidato,
            'conhecimentos_tecnicos_candidato': conhecimentos_tecnicos_candidato, 'titulo_candidato': titulo_candidato,
            'local_candidato': local_candidato, 'nivel_espanhol_candidato': 'Nenhum', 
        }])
        
        df_processado = criar_features(dados_input)
        vetor_X = criar_vetor_final(df_processado, sbert_model)
        probabilidade = pipeline.predict_proba(vetor_X)[0][1]
        score = probabilidade * 100
        
        st.markdown("---")
        st.subheader("Resultado da Análise")
        
        if probabilidade >= limiar_escolhido:
            st.success(f"✔️ BOM MATCH! (Score de Afinidade: {score:.1f}%)")
        else:
            st.warning(f"❌ MATCH BAIXO. (Score de Afinidade: {score:.1f}%)")
        
        st.progress(int(score))