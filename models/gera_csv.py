import json
import os
import polars as pl
import pandas as pd # Usado apenas para obter a lista de status

print("Iniciando o script de consolidação de dados...")

# --- Configuração de Arquivos ---
class Config:
    RAW_DATA_PATH = 'data/raw/'
    MODEL_PATH = 'models/'
    
    # Arquivos de entrada
    JOBS_FILE = os.path.join(RAW_DATA_PATH, 'jobs.json')
    APPLICANTS_FILE = os.path.join(RAW_DATA_PATH, 'applicants.json')
    PROSPECTS_FILE = os.path.join(RAW_DATA_PATH, 'prospects.json')
    
    # Arquivo de saída
    OUTPUT_CSV_FILE = 'dados_consolidados_analise.csv'
    
    # Status para definir o alvo (copiado de modelo.py)
    STATUS_POSITIVOS = {'contratado pela decision', 'contratado como hunting', 'aprovado', 'proposta aceita'}
    STATUS_NEGATIVOS = {'não aprovado pelo cliente', 'não aprovado pelo rh', 'desistiu', 'recusado', 'reprovado', 'não aprovado pelo requisitante'}

def gerar_csv_consolidado(config: Config):
    """
    Carrega, une e filtra os dados brutos, salvando o resultado em um único CSV.
    """
    try:
        print(f"Carregando dados de '{config.JOBS_FILE}', '{config.APPLICANTS_FILE}', e '{config.PROSPECTS_FILE}'...")
        jobs_data = json.load(open(config.JOBS_FILE, 'r', encoding='utf-8'))
        applicants_data = json.load(open(config.APPLICANTS_FILE, 'r', encoding='utf-8'))
        prospects_data = json.load(open(config.PROSPECTS_FILE, 'r', encoding='utf-8'))
        print("Dados JSON carregados com sucesso.")
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado - {e}")
        print("Verifique se a pasta 'data/raw' existe no local correto e contém os arquivos .json.")
        return

    # Conversão para DataFrames Polars
    print("Convertendo dados para DataFrames Polars...")
    df_jobs = pl.from_dicts([{'vaga_id': k, **v.get("informacoes_basicas", {}), **v.get("perfil_vaga", {})} for k, v in jobs_data.items()])
    if 'nivel profissional' in df_jobs.columns:
        df_jobs = df_jobs.rename({'nivel profissional': 'nivel_profissional_vaga', 'nivel_ingles': 'nivel_ingles_vaga', 'nivel_espanhol': 'nivel_espanhol_vaga', 'nivel_academico': 'nivel_academico_vaga'})
    
    applicant_rows = [{'candidato_id': cid, 'cv_pt': data.get('cv_pt'), **data.get('infos_basicas', {}), **data.get('formacao_e_idiomas', {}), **data.get('informacoes_profissionais', {})} for cid, data in applicants_data.items()]
    df_applicants = pl.from_dicts(applicant_rows)
    df_applicants = df_applicants.rename({'nivel_profissional': 'nivel_profissional_candidato', 'nivel_ingles': 'nivel_ingles_candidato', 'nivel_espanhol': 'nivel_espanhol_candidato', 'nivel_academico': 'nivel_academico_candidato'})
    
    prospect_rows = [{'vaga_id': k, 'candidato_id': p.get('codigo'), **p} for k, v in prospects_data.items() for p in v.get('prospects', [])]
    df_prospects = pl.from_dicts(prospect_rows)
    
    # Junção dos DataFrames
    print("Juntando os DataFrames de vagas, candidatos e prospects...")
    # Lógica de join corrigida para evitar erro de colunas duplicadas
    df_merged_vaga = df_prospects.join(df_jobs, on='vaga_id', how='left', suffix='_vaga')
    df_merged = df_merged_vaga.join(df_applicants, on='candidato_id', how='left', suffix='_candidato')
    print(f"Total de registros antes da filtragem: {df_merged.height}")

    # Filtragem para processos encerrados e CVs válidos
    print("Filtrando para manter apenas processos encerrados (status positivo ou negativo)...")
    df_final = df_merged.with_columns(
        pl.when(pl.col('situacao_candidado').str.to_lowercase().is_in(config.STATUS_POSITIVOS)).then(pl.lit(1))
        .when(pl.col('situacao_candidado').str.to_lowercase().is_in(config.STATUS_NEGATIVOS)).then(pl.lit(0))
        .otherwise(pl.lit(-1)) # Marca processos não encerrados
        .alias('alvo')
    ).filter(
        pl.col('alvo') != -1 # Mantém apenas os encerrados
    ).filter(
        pl.col('cv_pt').is_not_null() & (pl.col('cv_pt').str.len_bytes() > 1) # Mantém apenas com CV válido
    )
    print(f"Total de registros após a filtragem (processos encerrados): {df_final.height}")

    # Salvando em CSV
    try:
        print(f"Salvando o arquivo consolidado em '{config.OUTPUT_CSV_FILE}'...")
        df_final.write_csv(config.OUTPUT_CSV_FILE)
        print("\n--- SUCESSO! ---")
        print(f"Arquivo '{config.OUTPUT_CSV_FILE}' foi gerado com sucesso na pasta do projeto.")
        print("Por favor, envie este arquivo para análise.")
    except Exception as e:
        print(f"ERRO ao salvar o arquivo CSV: {e}")

if __name__ == "__main__":
    config = Config()
    gerar_csv_consolidado(config)