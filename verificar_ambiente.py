#!/usr/bin/env python3
"""
Script de verificação do ambiente para o projeto Decision
"""

def verificar_ambiente():
    print("🔍 Verificando ambiente do projeto Decision...\n")
    
    # Verificar Python
    import sys
    print(f"✅ Python: {sys.version}")
    
    # Verificar dependências principais
    try:
        import streamlit as st
        print(f"✅ Streamlit: {st.__version__}")
    except ImportError:
        print("❌ Streamlit não encontrado")
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError:
        print("❌ Pandas não encontrado")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy não encontrado")
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn não encontrado")
    
    try:
        import xgboost as xgb
        print(f"✅ XGBoost: {xgb.__version__}")
    except ImportError:
        print("❌ XGBoost não encontrado")
    
    # Verificar spaCy e modelo
    try:
        import spacy
        print(f"✅ spaCy: {spacy.__version__}")
        
        try:
            nlp = spacy.load("pt_core_news_sm")
            print("✅ Modelo spaCy português carregado com sucesso")
        except OSError:
            try:
                import pt_core_news_sm
                print("✅ Modelo spaCy português disponível via pacote")
            except ImportError:
                print("⚠️ Modelo spaCy português não encontrado - usando fallback")
    except ImportError:
        print("❌ spaCy não encontrado")
    
    # Verificar NLTK
    try:
        import nltk
        print(f"✅ NLTK: {nltk.__version__}")
        
        try:
            from nltk.corpus import stopwords
            stopwords.words('portuguese')
            print("✅ Stopwords em português disponíveis")
        except:
            print("⚠️ Stopwords em português não encontradas")
    except ImportError:
        print("❌ NLTK não encontrado")
    
    # Verificar modelos
    import os
    modelos_path = "models"
    if os.path.exists(modelos_path):
        modelos = [f for f in os.listdir(modelos_path) if f.endswith('.pkl')]
        if modelos:
            print(f"✅ Modelos encontrados: {len(modelos)}")
            for modelo in modelos:
                print(f"   - {modelo}")
        else:
            print("⚠️ Nenhum modelo .pkl encontrado na pasta models/")
    else:
        print("❌ Pasta models/ não encontrada")
    
    # Verificar dados
    dados_path = "data/raw"
    if os.path.exists(dados_path):
        dados = os.listdir(dados_path)
        if dados:
            print(f"✅ Dados encontrados: {len(dados)} arquivo(s)")
        else:
            print("⚠️ Pasta data/raw/ vazia")
    else:
        print("❌ Pasta data/raw/ não encontrada")
    
    print("\n🎉 Verificação concluída!")

if __name__ == "__main__":
    verificar_ambiente()
