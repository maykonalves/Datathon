import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import json
import os
from collections import Counter
import spacy
import nltk
from nltk.corpus import stopwords

# ==============================================================================
# 1. CONFIGURAÇÃO E CARREGAMENTO DOS MODELOS
# ==============================================================================

@st.cache_resource
def carregar_recursos_do_modelo():
    """
    Carrega o manifesto e todos os modelos especialistas e o geral disponíveis.
    Retorna um dicionário de modelos carregados e o manifesto.
    """
    model_path = 'models/'
    manifest_path = os.path.join(model_path, 'models_manifest.json')

    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except FileNotFoundError:
        st.error(f"Arquivo de manifesto '{manifest_path}' não encontrado. Execute o script de treinamento 'modelo.py' primeiro.")
        st.stop()

    loaded_models = {}
    
    # Carregar modelos especialistas
    for skill_name in manifest.get('specialist_models', []):
        try:
            model_file = os.path.join(model_path, f'modelo_{skill_name}.pkl')
            loaded_models[skill_name] = joblib.load(model_file)
            print(f"Modelo especialista '{skill_name}' carregado com sucesso.")
        except FileNotFoundError:
            st.warning(f"AVISO: Modelo para '{skill_name}' listado no manifesto mas não encontrado em '{model_file}'.")

    # Carregar modelo de fallback
    fallback_model_name = manifest.get('fallback_model')
    if fallback_model_name:
        try:
            model_file = os.path.join(model_path, f'modelo_{fallback_model_name}.pkl')
            loaded_models[fallback_model_name] = joblib.load(model_file)
            print(f"Modelo de fallback '{fallback_model_name}' carregado com sucesso.")
        except FileNotFoundError:
            st.error(f"ERRO CRÍTICO: Modelo de fallback '{fallback_model_name}' não encontrado.")
            st.stop()
            
    try:
        # Primeira tentativa: carregar modelo normalmente
        nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
    except OSError:
        try:
            # Segunda tentativa: carregar modelo pelo nome do pacote
            import pt_core_news_sm
            nlp = pt_core_news_sm.load(disable=["parser", "ner"])
        except (ImportError, OSError):
            try:
                # Terceira tentativa: download programático (pode falhar em produção)
                st.warning("⚠️ Tentando baixar modelo spaCy português...")
                import subprocess
                import sys
                result = subprocess.run([
                    sys.executable, "-m", "spacy", "download", "pt_core_news_sm"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
                else:
                    raise OSError("Falha no download do modelo")
            except Exception as e:
                # Fallback: usar modelo de linguagem básico
                st.warning(f"⚠️ Não foi possível carregar o modelo spaCy português. Usando processamento básico. Erro: {str(e)}")
                try:
                    # Tentar modelo em inglês como fallback
                    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                    st.info("✅ Usando modelo spaCy em inglês como fallback.")
                except OSError:
                    # Último recurso: modelo vazio
                    nlp = spacy.blank("pt")
                    st.warning("⚠️ Usando modelo spaCy básico sem recursos avançados.")

    return loaded_models, manifest, nlp

modelos, manifest, nlp = carregar_recursos_do_modelo()
nltk.download('stopwords', quiet=True)
STOPWORDS_PT = set(stopwords.words('portuguese'))

# Configurações do modelo replicadas para consistência
class ModelConfig:
    SKILL_HIERARCHY = {
        'python': ['python', 'django', 'flask', 'pyspark', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'fastapi'],
        'java': ['java', 'spring', 'springboot', 'hibernate', 'maven', 'gradle', 'jvm'],
        'sql': ['sql', 'postgres', 'postgresql', 'mysql', 't-sql', 'tsql', 'pl/sql', 'oracle sql', 'database', 'banco de dados', 'nosql', 'mongodb', 'cassandra'],
        'aws': ['aws', 'amazon web services', 's3', 'ec2', 'lambda', 'rds', 'dynamodb', 'cloudformation', 'eks', 'ecs', 'vpc', 'iam', 'cloudwatch'],
        'azure': ['azure', 'microsoft azure', 'azure devops', 'azure functions', 'azure ad', 'azure kubernetes service'],
        'sap': ['sap', 'abap', 'fiori', 'hana', 'sapui5', 'mm', 'sd', 'fi', 's/4hana', 'erp'],
        'react': ['react', 'reactjs', 'react.js', 'redux', 'next.js'],
        'angular': ['angular', 'angularjs'],
        'powerbi': ['powerbi', 'power bi', 'tableau', 'qlik sense'],
        'docker': ['docker', 'docker-compose', 'containerization'],
        'kubernetes': ['kubernetes', 'k8s', 'openshift', 'orchestration'],
        'oracle': ['oracle', 'oracle database', 'pl/sql', 'oracle cloud'],
    }
    TOP_SKILLS_FOR_FEATURES = list(SKILL_HIERARCHY.keys())

config = ModelConfig()

# ==============================================================================
# 2. FUNÇÕES DE FEATURE ENGINEERING (IDÊNTICAS AO `modelo.py`)
# ==============================================================================
# (Estas funções permanecem as mesmas da versão anterior)
MAPA_NIVEL_ACADEMICO = {'Não Informado': 0, 'Ensino Fundamental Incompleto': 1, 'Ensino Fundamental Cursando': 2, 'Ensino Fundamental Completo': 3, 'Ensino Médio Incompleto': 4, 'Ensino Médio Cursando': 5, 'Ensino Médio Completo': 6, 'Ensino Técnico Incompleto': 7, 'Ensino Técnico Cursando': 8, 'Ensino Técnico Completo': 9, 'Ensino Superior Incompleto': 10, 'Ensino Superior Cursando': 11, 'Ensino Superior Completo': 12, 'Pós Graduação Incompleto': 13, 'Pós Graduação Cursando': 14, 'Pós Graduação Completo': 15, 'Mestrado Incompleto': 16, 'Mestrado Cursando': 17, 'Mestrado Completo': 18, 'Doutorado Incompleto': 19, 'Doutorado Cursando': 20, 'Doutorado Completo': 21}
MAPA_NIVEL_PROFISSIONAL = {'Não Informado': 0, 'Aprendiz': 1, 'Auxiliar': 2, 'Assistente': 3, 'Trainee': 4, 'Júnior': 5, 'Técnico de Nível Médio': 6, 'Pleno': 7, 'Sênior': 8, 'Especialista': 9, 'Líder': 10, 'Supervisor': 11, 'Coordenador': 12, 'Gerente': 13, 'Outro': 0}
MAPA_NIVEL_IDIOMA = {'Não Informado': 0, 'Nenhum': 0, 'Básico': 1, 'Técnico': 2, 'Intermediário': 3, 'Avançado': 4, 'Fluente': 5}

def mapear_nivel_academico_pd(series: pd.Series) -> pd.Series: return series.map(MAPA_NIVEL_ACADEMICO).fillna(0).astype('int8')
def mapear_nivel_profissional_pd(series: pd.Series) -> pd.Series: return series.map(MAPA_NIVEL_PROFISSIONAL).fillna(0).astype('int8')
def obter_nivel_idioma_pd(series: pd.Series) -> pd.Series: return series.str.title().map(MAPA_NIVEL_IDIOMA).fillna(0).astype('int8')
def calcular_match_trinario(nivel_candidato: pd.Series, nivel_vaga: pd.Series) -> pd.Series:
    conditions = [(nivel_vaga == 0), (nivel_candidato >= nivel_vaga), (nivel_candidato < nivel_vaga)]
    choices = [0, 1, -1]
    return np.select(conditions, choices, default=0).astype('int8')
def calcular_diferenca_nivel(nivel_candidato: pd.Series, nivel_vaga: pd.Series) -> pd.Series: return np.where(nivel_vaga == 0, 0, nivel_candidato - nivel_vaga).astype('int8')

def _get_skill_principal(texto_vaga: str, skill_hierarchy: dict) -> str:
    if not isinstance(texto_vaga, str): return 'outra'
    texto_vaga = texto_vaga.lower()
    contagem_skills = Counter()
    for canonical_skill, aliases in skill_hierarchy.items():
        pattern = r'\b(' + '|'.join(re.escape(alias) for alias in aliases) + r')\b'
        matches = re.findall(pattern, texto_vaga)
        if matches: contagem_skills[canonical_skill] += len(matches)
    if not contagem_skills: return 'outra'
    return contagem_skills.most_common(1)[0][0]

def criar_features_para_previsao(df_pd: pd.DataFrame) -> (pd.DataFrame, str):
    df = df_pd.copy().fillna("")
    
    # Lógica de criação de features omitida por brevidade (é a mesma da versão anterior de `modelo.py`)
    df['match_nivel_academico'] = calcular_match_trinario(mapear_nivel_academico_pd(df['nivel_academico_candidato']), mapear_nivel_academico_pd(df['nivel_academico_vaga']))
    df['match_nivel_profissional'] = calcular_match_trinario(mapear_nivel_profissional_pd(df['nivel_profissional_candidato']), mapear_nivel_profissional_pd(df['nivel_profissional_vaga']))
    df['match_ingles'] = calcular_match_trinario(obter_nivel_idioma_pd(df['nivel_ingles_candidato']), obter_nivel_idioma_pd(df['nivel_ingles_vaga']))
    df['match_espanhol'] = calcular_match_trinario(obter_nivel_idioma_pd(df['nivel_espanhol_candidato']), obter_nivel_idioma_pd(df['nivel_espanhol_vaga']))
    df['diff_nivel_academico'] = calcular_diferenca_nivel(mapear_nivel_academico_pd(df['nivel_academico_candidato']), mapear_nivel_academico_pd(df['nivel_academico_vaga']))
    df['diff_nivel_profissional'] = calcular_diferenca_nivel(mapear_nivel_profissional_pd(df['nivel_profissional_candidato']), mapear_nivel_profissional_pd(df['nivel_profissional_vaga']))
    df['diff_ingles'] = calcular_diferenca_nivel(obter_nivel_idioma_pd(df['nivel_ingles_candidato']), obter_nivel_idioma_pd(df['nivel_ingles_vaga']))
    df['diff_espanhol'] = calcular_diferenca_nivel(obter_nivel_idioma_pd(df['nivel_espanhol_candidato']), obter_nivel_idioma_pd(df['nivel_espanhol_vaga']))
    texto_vaga_full = (df['titulo_vaga'] + ' ' + df['principais_atividades'] + ' ' + df['competencia_tecnicas_e_comportamentais']).str.lower()
    texto_candidato_full = (df['cv_pt'] + ' ' + df['objetivo_profissional'] + ' ' + df['comentario']).str.lower()
    for skill, aliases in config.SKILL_HIERARCHY.items():
        pattern = r'\b(' + '|'.join(re.escape(alias) for alias in aliases) + r')\b'
        df[f'match_skill_{skill}'] = calcular_match_trinario(texto_candidato_full.str.contains(pattern, regex=True).astype(int), texto_vaga_full.str.contains(pattern, regex=True).astype(int))
        df[f'count_skill_vaga_{skill}'] = texto_vaga_full.str.count(pattern)
        df[f'count_skill_cand_{skill}'] = texto_candidato_full.str.count(pattern)
    texto_vaga_identificacao = df['titulo_vaga'] + ' ' + df['principais_atividades']
    skill_principal = _get_skill_principal(texto_vaga_identificacao.iloc[0], config.SKILL_HIERARCHY)
    df['match_skill_principal'] = df.apply(lambda row: row.get(f"match_skill_{skill_principal}", 0), axis=1)
    df['contagem_skill_principal_vaga'] = df.apply(lambda row: row.get(f"count_skill_vaga_{skill_principal}", 0), axis=1)
    df['contagem_skill_principal_candidato'] = df.apply(lambda row: row.get(f"count_skill_cand_{skill_principal}", 0), axis=1)

    feature_cols = [
        'match_nivel_academico', 'match_nivel_profissional', 'match_ingles', 'match_espanhol',
        'diff_nivel_academico', 'diff_nivel_profissional', 'diff_ingles', 'diff_espanhol',
        'match_skill_principal', 'contagem_skill_principal_vaga', 'contagem_skill_principal_candidato'
    ] + [f'match_skill_{s}' for s in config.TOP_SKILLS_FOR_FEATURES] \
      + [f'count_skill_vaga_{s}' for s in config.TOP_SKILLS_FOR_FEATURES] \
      + [f'count_skill_cand_{s}' for s in config.TOP_SKILLS_FOR_FEATURES]
    X = df[feature_cols].fillna(0)
    return X, skill_principal

# ==============================================================================
# INTERFACE DO STREAMLIT
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análise de Match")
st.title("Análise de Match: Vaga vs. Candidato")
st.markdown("Insira as informações da vaga e do candidato para calcular a probabilidade de sucesso.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Informações da Vaga")
    titulo_vaga = st.text_input("Título da Vaga", "Desenvolvedor Python Sênior")
    principais_atividades = st.text_area("Principais Atividades", "Desenvolvimento de APIs REST com Django e Flask.", height=125)
    competencia_tecnicas = st.text_area("Competências", "Experiência com AWS e Docker. Conhecimento em SQL e bancos de dados PostgreSQL.", height=125)
    nivel_profissional_vaga = st.selectbox("Nível Profissional Vaga", list(MAPA_NIVEL_PROFISSIONAL.keys()), index=8)
    nivel_academico_vaga = st.selectbox("Nível Acadêmico Vaga", list(MAPA_NIVEL_ACADEMICO.keys()), index=12)
    nivel_ingles_vaga = st.selectbox("Nível de Inglês Vaga", list(MAPA_NIVEL_IDIOMA.keys()), index=4)

with col2:
    st.subheader("Informações do Candidato")
    objetivo_profissional = st.text_input("Objetivo Profissional", "Atuar como Desenvolvedor Python Sênior")
    cv_pt = st.text_area("Currículo", "Vasta experiência como desenvolvedor Python, criando soluções escaláveis na nuvem (AWS). Domínio de PostgreSQL e desenvolvimento de microserviços com Docker e Kubernetes.", height=125)
    comentario_candidato = st.text_area("Comentários Adicionais", "Disponibilidade para início imediato.", height=125)
    nivel_profissional_candidato = st.selectbox("Nível Profissional Candidato", list(MAPA_NIVEL_PROFISSIONAL.keys()), index=8)
    nivel_academico_candidato = st.selectbox("Nível Acadêmico Candidato", list(MAPA_NIVEL_ACADEMICO.keys()), index=15)
    nivel_ingles_candidato = st.selectbox("Nível de Inglês Candidato", list(MAPA_NIVEL_IDIOMA.keys()), index=5)

st.sidebar.title("Configurações")
limiar_escolhido = st.sidebar.slider("Limiar de Decisão", 0.0, 1.0, 0.5, 0.01)
st.sidebar.info(f"Candidatos com score > {limiar_escolhido*100:.0f}% serão 'Bom Match'.")

if st.button("Calcular Match", type="primary", use_container_width=True):
    with st.spinner('Analisando perfil e calculando score...'):
        dados_input = pd.DataFrame([{'titulo_vaga': titulo_vaga, 'principais_atividades': principais_atividades,
            'competencia_tecnicas_e_comportamentais': competencia_tecnicas, 'objetivo_profissional': objetivo_profissional,
            'cv_pt': cv_pt, 'comentario': comentario_candidato, 'nivel_profissional_vaga': nivel_profissional_vaga,
            'nivel_academico_vaga': nivel_academico_vaga, 'nivel_ingles_vaga': nivel_ingles_vaga, 'nivel_espanhol_vaga': 'Nenhum',
            'nivel_profissional_candidato': nivel_profissional_candidato, 'nivel_academico_candidato': nivel_academico_candidato,
            'nivel_ingles_candidato': nivel_ingles_candidato, 'nivel_espanhol_candidato': 'Nenhum'}])

        features, skill_principal_identificada = criar_features_para_previsao(dados_input)
        
        # --- LÓGICA DE SELEÇÃO DINÂMICA DE MODELO ---
        modelo_selecionado = None
        modelo_usado_nome = ""

        if skill_principal_identificada in manifest['specialist_models']:
            modelo_selecionado = modelos[skill_principal_identificada]
            modelo_usado_nome = f"Especialista: {skill_principal_identificada.upper()}"
        else:
            modelo_selecionado = modelos[manifest['fallback_model']]
            modelo_usado_nome = f"Geral (Fallback)"
            
        st.info(f"**Skill Principal da Vaga:** `{skill_principal_identificada.upper()}` → **Usando Modelo:** `{modelo_usado_nome}`")

        probabilidade = modelo_selecionado.predict_proba(features)[0][1]
        score = probabilidade * 100
        
        st.markdown("---")
        st.subheader("Resultado da Análise")
        
        if probabilidade >= limiar_escolhido:
            st.success(f"✔️ BOM MATCH! (Score de Afinidade: {score:.1f}%)")
        else:
            st.warning(f"❌ MATCH BAIXO. (Score de Afinidade: {score:.1f}%)")
        
        st.progress(int(score))

        with st.expander("Ver Análise Detalhada dos Pontos"):
            st.markdown("##### Análise de Requisitos Principais")
            col_detalhes1, col_detalhes2, col_detalhes3 = st.columns(3)
            col_detalhes1.metric(label="Match Skill Principal", value=str(features['match_skill_principal'].iloc[0]))
            col_detalhes2.metric(label="Match Nível Profissional", value=str(features['match_nivel_profissional'].iloc[0]))
            col_detalhes3.metric(label="Match Inglês", value=str(features['match_ingles'].iloc[0]))
            st.caption("Legenda Match: 1 = Atende, 0 = Não Exigido, -1 = Não Atende")

            st.markdown("##### Análise de Todas as Skills")
            skills_match_cols = [col for col in features.columns if col.startswith('match_skill_') and col != 'match_skill_principal']
            df_skills_details = pd.DataFrame(index=[s.replace('match_skill_', '') for s in skills_match_cols])
            df_skills_details['Status Match'] = features[skills_match_cols].iloc[0].map({1: '✅ Atende', 0: '➖ Neutro', -1: '❌ Não Atende'}).values
            for s in df_skills_details.index:
                df_skills_details.loc[s, 'Menções Vaga'] = features.get(f'count_skill_vaga_{s}', pd.Series([0])).iloc[0]
                df_skills_details.loc[s, 'Menções Candidato'] = features.get(f'count_skill_cand_{s}', pd.Series([0])).iloc[0]
            st.dataframe(df_skills_details.astype({'Menções Vaga': int, 'Menções Candidato': int}))