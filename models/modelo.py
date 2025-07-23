import pandas as pd
import numpy as np
import re
import json
import os
import joblib

from unidecode import unidecode
from nltk.corpus import stopwords
import nltk
import unicodedata

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

# Baixar os stopwords do NLTK (necessário na primeira execução)
try:
    stopwords.words('portuguese')
except LookupError:
    print("Baixando stopwords do NLTK...")
    nltk.download('stopwords')

class MatchModel:
    """
    Uma classe para treinar, avaliar e usar um modelo de machine learning
    para prever o match entre candidatos e vagas.
    """
    def __init__(self, model_path='models'):
        self.model_path = model_path
        self.pipeline = None
        self.sbert_model = None
        self.threshold = 0.20  # Limiar padrão

        # Constantes e Mapeamentos
        self.STATUS_POSITIVOS = {'contratado pela decision', 'contratado como hunting', 'aprovado', 'proposta aceita'}
        self.STATUS_NEGATIVOS = {'não aprovado pelo cliente', 'não aprovado pelo rh', 'desistiu', 'recusado', 'reprovado', 'não aprovado pelo requisitante'}
        
        self.MAPA_IDIOMAS = {
            'Nenhum': 0, 'Básico': 1, 'Técnico': 2, 'Intermediário': 3,
            'Avançado': 4, 'Fluente': 5
        }

        self.HIERARQUIA_CARGOS = {
            'aprendiz': 1, 'trainee': 2, 'auxiliar': 3, 'assistente': 4,
            'analista': 5, 'junior': 6, 'pleno': 7, 'senior': 8,
            'especialista': 9, 'supervisor': 10, 'lider': 11,
            'coordenador': 12, 'gerente': 13
        }
        
        self.STOP_WORDS = set(stopwords.words('portuguese'))

    def _load_json(self, path):
        """Carrega um arquivo JSON."""
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def _preprocess_text(self, text):
        """Limpa e pré-processa o texto para NLP."""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.STOP_WORDS]
        return ' '.join(tokens)

    def _get_seniority_level(self, text):
        """Extrai o maior nível de senioridade de um texto."""
        palavras_chave = self.HIERARQUIA_CARGOS.keys()
        niveis_encontrados = [p for p in palavras_chave if p in text.split()]
        if not niveis_encontrados:
            return None
        return max(niveis_encontrados, key=lambda x: self.HIERARQUIA_CARGOS[x])

    def _feature_engineering(self, df):
        """Aplica toda a engenharia de features no DataFrame."""
        # Combinação de textos
        df['texto_candidato'] = (
            df['cv'].fillna('') + ' ' + df['conhecimentos_tecnicos_candidato'].fillna('') + ' ' +
            df['nivel_profissional_candidato'].fillna('') + ' ' + df['titulo_candidato'].fillna('') + ' ' +
            df['objetivo_candidato'].fillna('')
        )
        df['texto_vaga'] = (
            df['titulo_vaga'].fillna('') + ' ' + df['area_atuacao_vaga'].fillna('') + ' ' +
            df['principais_atividades_vaga'].fillna('') + ' ' + df['competencias_tecnicas_vaga'].fillna('')
        )
        df['texto_vaga_tratado'] = df['texto_vaga'].apply(self._preprocess_text)
        df['texto_candidato_tratado'] = df['texto_candidato'].apply(self._preprocess_text)
        
        # Idiomas
        for col in ['nivel_ingles_vaga', 'nivel_espanhol_vaga', 'nivel_ingles_candidato', 'nivel_espanhol_candidato']:
            df[col] = df[col].replace('', 'Nenhum').fillna('Nenhum').str.strip().str.title()
            df[col + '_num'] = df[col].map(self.MAPA_IDIOMAS)
        
        df['match_nivel_ingles'] = (df['nivel_ingles_candidato_num'] >= df['nivel_ingles_vaga_num']).astype(int)
        df['match_nivel_espanhol'] = (df['nivel_espanhol_candidato_num'] >= df['nivel_espanhol_vaga_num']).astype(int)

        # Localização
        df['local_vaga_tratado'] = df['local_vaga'].apply(lambda x: unidecode(str(x)).lower().strip() if pd.notna(x) else '')
        df['local_candidato_tratado'] = df['local_candidato'].apply(lambda x: unidecode(str(x).split(',')[0]).lower().strip() if pd.notna(x) else '')
        df['match_local_cidade_bin'] = (df['local_vaga_tratado'] == df['local_candidato_tratado']).astype(int)
        
        # Nível de Senioridade
        df['nivel_profissional_vaga'] = df['nivel_profissional_vaga'].apply(lambda x: unidecode(str(x)).lower().strip() if pd.notna(x) else None)
        df['maior_nivel_detectado'] = df['texto_candidato_tratado'].apply(self._get_seniority_level)
        
        def comparar_niveis(row):
            detectado = row['maior_nivel_detectado']
            real = row['nivel_profissional_vaga']
            if pd.isna(detectado) or pd.isna(real) or real not in self.HIERARQUIA_CARGOS:
                return 0 # sem_info ou abaixo
            return 1 if self.HIERARQUIA_CARGOS.get(detectado, 0) >= self.HIERARQUIA_CARGOS.get(real, 0) else 0

        df['match_nivel_bin'] = df.apply(comparar_niveis, axis=1)

        def calcular_diferenca(row):
            detectado = row['maior_nivel_detectado']
            real = row['nivel_profissional_vaga']
            if pd.isna(detectado) or pd.isna(real) or real not in self.HIERARQUIA_CARGOS:
                return 0
            return self.HIERARQUIA_CARGOS.get(detectado, 0) - self.HIERARQUIA_CARGOS.get(real, 0)

        df['diferenca_nivel_senioridade'] = df.apply(calcular_diferenca, axis=1)

        return df

    def _get_feature_vector(self, df):
        """Cria o vetor de features final (X)."""
        if not self.sbert_model:
            self.sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            
        X_vaga = self.sbert_model.encode(df['texto_vaga_tratado'].tolist(), show_progress_bar=False)
        X_candidato = self.sbert_model.encode(df['texto_candidato_tratado'].tolist(), show_progress_bar=False)
        
        features_adicionais = df[[
            'match_nivel_bin', 'match_nivel_ingles', 'match_nivel_espanhol',
            'diferenca_nivel_senioridade', 'match_local_cidade_bin'
        ]].values
        
        return np.hstack((X_vaga, X_candidato, features_adicionais))

    def train(self, jobs_path, applicants_path, prospects_path):
        """
        Orquestra o processo completo de carregamento, pré-processamento e treinamento do modelo.
        """
        print("Iniciando o processo de treinamento...")
        
        # 1. Carregar e mesclar dados
        print("1/5 - Carregando e mesclando dados...")
        jobs_data = self._load_json(jobs_path)
        applicants_data = self._load_json(applicants_path)
        prospects_data = self._load_json(prospects_path)

        df_jobs = pd.DataFrame([{'vaga_id': k, **v.get("informacoes_basicas", {}), **v.get("perfil_vaga", {})} for k, v in jobs_data.items()])
        
        # <<< CORREÇÃO: Ajuste na criação do DataFrame de candidatos para incluir o 'cv_pt'
        df_applicants = pd.DataFrame([
            {
                'candidato_id': k,
                'cv_pt': v.get('cv_pt'),  # Adiciona a chave 'cv_pt' que estava faltando
                **v.get("infos_basicas", {}),
                **v.get("formacao_e_idiomas", {}),
                **v.get("informacoes_profissionais", {})
            }
            for k, v in applicants_data.items()
        ])
        
        prospect_rows = [{'vaga_id': v_id, 'candidato_id': p.get('codigo'), **p} for v_id, v_info in prospects_data.items() for p in v_info.get('prospects', [])]
        df_prospects = pd.DataFrame(prospect_rows)

        df_merged = df_prospects.merge(df_jobs, on='vaga_id', how='left', suffixes=('', '_vaga'))
        df_merged = df_merged.merge(df_applicants, on='candidato_id', how='left', suffixes=('', '_candidato'))
        
        if 'situacao_candidado' in df_merged.columns:
            df_merged.rename(columns={'situacao_candidado': 'situacao'}, inplace=True)

        # Renomeando colunas para consistência
        df_merged.rename(columns={
            'titulo_vaga': 'titulo_vaga', 'tipo_contratacao': 'tipo_contratacao_vaga',
            'nivel profissional': 'nivel_profissional_vaga', 'nivel_ingles': 'nivel_ingles_vaga',
            'nivel_espanhol': 'nivel_espanhol_vaga', 'areas_atuacao': 'area_atuacao_vaga',
            'cidade': 'local_vaga', 'principais_atividades': 'principais_atividades_vaga',
            'competencia_tecnicas_e_comportamentais': 'competencias_tecnicas_vaga',
            'local': 'local_candidato', 'nivel_ingles_candidato': 'nivel_ingles_candidato',
            'nivel_espanhol_candidato': 'nivel_espanhol_candidato', 'nivel_academico': 'nivel_academico_candidato',
            'area_atuacao': 'area_atuacao_candidato', 'nivel_profissional': 'nivel_profissional_candidato',
            'conhecimentos_tecnicos': 'conhecimentos_tecnicos_candidato', 'titulo_profissional': 'titulo_candidato',
            'objetivo_profissional': 'objetivo_candidato', 'cv_pt': 'cv' # Esta linha agora funciona corretamente
        }, inplace=True, errors='ignore')

        # 2. Criar variável alvo e filtrar dados
        print("2/5 - Criando variável alvo e filtrando dados...")
        status_a_manter = self.STATUS_POSITIVOS.union(self.STATUS_NEGATIVOS)
        df_merged['situacao_lower'] = df_merged['situacao'].str.lower()
        df_merged = df_merged[df_merged['situacao_lower'].isin(status_a_manter)].copy()
        df_merged['alvo'] = np.where(df_merged['situacao_lower'].isin(self.STATUS_POSITIVOS), 1, 0)
        df_merged.drop(columns=['situacao_lower'], inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

        # 3. Engenharia de Features
        print("3/5 - Aplicando engenharia de features...")
        df_processed = self._feature_engineering(df_merged)

        # 4. Criar Vetor de Features e Alvo (X, y)
        print("4/5 - Criando vetores para o modelo...")
        X = self._get_feature_vector(df_processed)
        y = df_processed['alvo'].values

        # 5. Treinamento do Modelo
        print("5/5 - Treinando o modelo XGBoost...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb = XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        
        self.pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', xgb)
        ])
        
        self.pipeline.fit(X_train_res, y_train_res)
        
        print("\n--- Avaliação do Modelo no Conjunto de Teste ---")
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        print(f"Relatório de Classificação (Limiar = {self.threshold}):")
        print(classification_report(y_test, y_pred))
        print("Treinamento concluído!")

    def predict(self, df_input):
        """
        Realiza a predição para um DataFrame de entrada.
        O DataFrame deve ter as mesmas colunas brutas do que foi usado no treino.
        """
        if not self.pipeline:
            raise RuntimeError("O modelo precisa ser treinado ou carregado antes da predição.")
        
        print("Aplicando pré-processamento e engenharia de features nos novos dados...")
        df_processed = self._feature_engineering(df_input)
        
        print("Criando vetor de features para predição...")
        X = self._get_feature_vector(df_processed)
        
        print("Realizando predição...")
        probabilities = self.pipeline.predict_proba(X)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        
        return predictions, probabilities

    def save_model(self):
        """Salva o pipeline treinado e o limiar."""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        pipeline_path = os.path.join(self.model_path, 'xgb_pipeline_treinado.pkl')
        threshold_path = os.path.join(self.model_path, 'limiar_xgb.txt')
        
        joblib.dump(self.pipeline, pipeline_path)
        with open(threshold_path, 'w') as f:
            f.write(str(self.threshold))
            
        print(f"Pipeline salvo em '{pipeline_path}'")
        print(f"Limiar ({self.threshold}) salvo em '{threshold_path}'")

    def load_model(self):
        """Carrega um pipeline e limiar previamente salvos."""
        pipeline_path = os.path.join(self.model_path, 'xgb_pipeline_treinado.pkl')
        threshold_path = os.path.join(self.model_path, 'limiar_xgb.txt')

        if not os.path.exists(pipeline_path) or not os.path.exists(threshold_path):
            raise FileNotFoundError(
                f"Arquivos de modelo não encontrados em '{self.model_path}'. "
                "Execute o script de treinamento primeiro."
            )

        self.pipeline = joblib.load(pipeline_path)
        with open(threshold_path, 'r') as f:
            self.threshold = float(f.read())
            
        # Garante que o modelo SBERT esteja carregado para predições
        if not self.sbert_model:
            self.sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            
        print(f"Pipeline e limiar ({self.threshold}) carregados com sucesso.")


if __name__ == '__main__':
    # Este bloco será executado quando você rodar o script diretamente
    # Exemplo de como treinar e salvar o modelo
    
    # Caminhos para os seus dados brutos
    JOBS_PATH = 'data/raw/jobs.json'
    APPLICANTS_PATH = 'data/raw/applicants.json'
    PROSPECTS_PATH = 'data/raw/prospects.json'
    
    # Instanciar o modelo
    model = MatchModel(model_path='models')
    
    # Treinar o modelo
    model.train(
        jobs_path=JOBS_PATH,
        applicants_path=APPLICANTS_PATH,
        prospects_path=PROSPECTS_PATH
    )
    
    # Salvar o modelo treinado
    model.save_model()