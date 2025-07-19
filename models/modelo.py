# -*- coding: utf-8 -*-
import json
import re
import time
import os
from collections import Counter

# --- Imports de PNL e ML ---
import spacy
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIGURAÇÃO E RECURSOS DE PNL
# ==============================================================================
# (Esta seção permanece inalterada)
nltk.download('stopwords', quiet=True)
STOPWORDS_PT = set(stopwords.words('portuguese'))
try:
    nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
except OSError:
    from spacy.cli import download
    download("pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])

class ModelConfig:
    PROCESSED_DATA_PATH = './'
    MODEL_PATH = 'models/'
    CONSOLIDATED_DATA_FILE = os.path.join(PROCESSED_DATA_PATH, 'dados_consolidados_analise.csv')
    MANIFEST_FILE = os.path.join(MODEL_PATH, 'models_manifest.json')
    
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
    MIN_SAMPLES_FOR_SPECIALIST_MODEL = 100 # Mínimo de vagas para treinar um modelo especialista

# ==============================================================================
# 2. FUNÇÕES AUXILIARES DE FEATURE ENGINEERING
# ==============================================================================
# (Esta seção permanece inalterada)
MAPA_NIVEL_ACADEMICO = {'Ensino Fundamental Incompleto': 1, 'Ensino Fundamental Cursando': 2, 'Ensino Fundamental Completo': 3, 'Ensino Médio Incompleto': 4, 'Ensino Médio Cursando': 5, 'Ensino Médio Completo': 6, 'Ensino Técnico Incompleto': 7, 'Ensino Técnico Cursando': 8, 'Ensino Técnico Completo': 9, 'Ensino Superior Incompleto': 10, 'Ensino Superior Cursando': 11, 'Ensino Superior Completo': 12, 'Pós Graduação Incompleto': 13, 'Pós Graduação Cursando': 14, 'Pós Graduação Completo': 15, 'Mestrado Incompleto': 16, 'Mestrado Cursando': 17, 'Mestrado Completo': 18, 'Doutorado Incompleto': 19, 'Doutorado Cursando': 20, 'Doutorado Completo': 21}
MAPA_NIVEL_PROFISSIONAL = {'Aprendiz': 1, 'Auxiliar': 2, 'Assistente': 3, 'Trainee': 4, 'Júnior': 5, 'Técnico de Nível Médio': 6, 'Pleno': 7, 'Sênior': 8, 'Especialista': 9, 'Líder': 10, 'Supervisor': 11, 'Coordenador': 12, 'Gerente': 13, 'Outro': 0}
MAPA_NIVEL_IDIOMA = {'Nenhum': 0, 'Básico': 1, 'Técnico': 2, 'Intermediário': 3, 'Avançado': 4, 'Fluente': 5}

def mapear_nivel_academico_pd(series: pd.Series) -> pd.Series: return series.map(MAPA_NIVEL_ACADEMICO).fillna(0).astype('int8')
def mapear_nivel_profissional_pd(series: pd.Series) -> pd.Series: return series.map(MAPA_NIVEL_PROFISSIONAL).fillna(0).astype('int8')
def obter_nivel_idioma_pd(series: pd.Series) -> pd.Series: return series.str.title().map(MAPA_NIVEL_IDIOMA).fillna(0).astype('int8')
def calcular_match_trinario(nivel_candidato: pd.Series, nivel_vaga: pd.Series) -> pd.Series:
    conditions = [(nivel_vaga == 0), (nivel_candidato >= nivel_vaga), (nivel_candidato < nivel_vaga)]
    choices = [0, 1, -1]
    return np.select(conditions, choices, default=0).astype('int8')
def calcular_diferenca_nivel(nivel_candidato: pd.Series, nivel_vaga: pd.Series) -> pd.Series: return np.where(nivel_vaga == 0, 0, nivel_candidato - nivel_vaga).astype('int8')

# ==============================================================================
# 3. MÓDULO DE CRIAÇÃO DE FEATURES
# ==============================================================================
# (Função _get_skill_principal e criar_features permanecem as mesmas da versão anterior)
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

def criar_features(df_pd: pd.DataFrame, config: 'ModelConfig') -> pd.DataFrame:
    print("\n--- Iniciando Criação de Features ---")
    df = df_pd.copy().fillna("")
    
    # Features de Nível, Diferença e Contagem de Skills
    # (Lógica omitida por brevidade, é a mesma da versão anterior)
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
    df['skill_principal_vaga'] = texto_vaga_identificacao.apply(lambda x: _get_skill_principal(x, config.SKILL_HIERARCHY))
    
    df['match_skill_principal'] = df.apply(lambda row: row.get(f"match_skill_{row['skill_principal_vaga']}", 0), axis=1)
    df['contagem_skill_principal_vaga'] = df.apply(lambda row: row.get(f"count_skill_vaga_{row['skill_principal_vaga']}", 0), axis=1)
    df['contagem_skill_principal_candidato'] = df.apply(lambda row: row.get(f"count_skill_cand_{row['skill_principal_vaga']}", 0), axis=1)

    print("Criação de features concluída.")
    return df

# ==============================================================================
# 4. MÓDULO DE TREINAMENTO DE UM ÚNICO MODELO (ESPECIALISTA OU GERAL)
# ==============================================================================
def train_single_model(X: pd.DataFrame, y: pd.Series, model_name: str, config: 'ModelConfig'):
    model_path = os.path.join(config.MODEL_PATH, f'modelo_{model_name}.pkl')
    shap_plot_path = os.path.join(config.MODEL_PATH, f'shap_summary_{model_name}.png')
    
    print(f"\n--- Treinando modelo para: '{model_name.upper()}' ({len(X)} amostras) ---")
    
    scale_pos_weight = y.value_counts().get(0, 1) / y.value_counts().get(1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Split de Treino/Teste: {len(X_train)}/{len(X_test)}. scale_pos_weight: {scale_pos_weight:.2f}")

    # Definindo o modelo e o grid de hiperparâmetros
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                              random_state=42, scale_pos_weight=scale_pos_weight)
    
    param_grid = {'max_depth': [5, 8], 'learning_rate': [0.1, 0.15],
                  'n_estimators': [150, 250], 'subsample': [0.7, 0.9]}
    
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=cv_strategy, verbose=0, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Melhor F1-score (CV): {grid_search.best_score_:.4f}")
    
    # Salvando o modelo treinado
    joblib.dump(best_model, model_path)
    print(f"Modelo salvo em: '{model_path}'")

    # Gerando e salvando o gráfico SHAP
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f'Importância das Features - Modelo {model_name.upper()}')
    plt.tight_layout()
    plt.savefig(shap_plot_path)
    plt.close()
    print(f"Gráfico SHAP salvo em: '{shap_plot_path}'")

# ==============================================================================
# 5. BLOCO DE EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    config = ModelConfig()

    try:
        # Carregar os dados consolidados
        df_full = pd.read_csv(config.CONSOLIDATED_DATA_FILE)

        # Criar todas as features de uma vez
        df_featured = criar_features(df_full, config)

        # Definir as colunas de features a serem usadas pelos modelos
        feature_cols = [
            'match_nivel_academico', 'match_nivel_profissional', 'match_ingles', 'match_espanhol',
            'diff_nivel_academico', 'diff_nivel_profissional', 'diff_ingles', 'diff_espanhol',
            'match_skill_principal', 'contagem_skill_principal_vaga', 'contagem_skill_principal_candidato'
        ] + [f'match_skill_{s}' for s in config.TOP_SKILLS_FOR_FEATURES] \
          + [f'count_skill_vaga_{s}' for s in config.TOP_SKILLS_FOR_FEATURES] \
          + [f'count_skill_cand_{s}' for s in config.TOP_SKILLS_FOR_FEATURES]
        
        # Identificar skills com dados suficientes para um modelo especialista
        skill_counts = df_featured['skill_principal_vaga'].value_counts()
        specialist_skills = skill_counts[skill_counts >= config.MIN_SAMPLES_FOR_SPECIALIST_MODEL].index.tolist()
        
        if 'outra' in specialist_skills:
            specialist_skills.remove('outra')
            
        print(f"\nSkills com dados suficientes para modelo especialista: {specialist_skills}")
        
        trained_models_list = []
        
        # Loop para treinar modelos especialistas
        for skill in specialist_skills:
            df_skill_subset = df_featured[df_featured['skill_principal_vaga'] == skill]
            X_subset = df_skill_subset[feature_cols]
            y_subset = df_skill_subset['alvo']
            
            # Checagem extra para garantir que temos ambas as classes (0 e 1)
            if len(y_subset.unique()) < 2:
                print(f"AVISO: Pulando skill '{skill}' por ter apenas uma classe de alvo.")
                continue
                
            train_single_model(X_subset, y_subset, model_name=skill, config=config)
            trained_models_list.append(skill)
            
        # Treinar modelo geral com os dados restantes
        df_general_subset = df_featured[~df_featured['skill_principal_vaga'].isin(specialist_skills)]
        if not df_general_subset.empty:
            X_general = df_general_subset[feature_cols]
            y_general = df_general_subset['alvo']
            if len(y_general.unique()) > 1:
                train_single_model(X_general, y_general, model_name='general', config=config)
                fallback_model_name = 'general'
            else:
                fallback_model_name = None
        
        # Salvar o manifesto de modelos
        manifest = {
            "specialist_models": trained_models_list,
            "fallback_model": fallback_model_name
        }
        with open(config.MANIFEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=4)
        print(f"\nManifesto de modelos salvo em '{config.MANIFEST_FILE}'")

    except FileNotFoundError:
        print(f"ERRO: Arquivo '{config.CONSOLIDATED_DATA_FILE}' não encontrado.")
    except Exception as e:
        import traceback
        print(f"\nERRO CRÍTICO no pipeline: {e}")
        traceback.print_exc()
    finally:
        elapsed = time.time() - start_time
        print(f"\n--- Pipeline de treinamento concluído em {elapsed:.2f} segundos. ---")