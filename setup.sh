#!/bin/bash

# Script de inicialização para Streamlit Cloud
echo "🚀 Iniciando configuração do ambiente..."

# Baixar recursos NLTK
python -c "
import nltk
print('📚 Baixando recursos NLTK...')
nltk.download('stopwords', quiet=True)
print('✅ Recursos NLTK baixados com sucesso!')
"

# Verificar modelo spaCy
python -c "
import spacy
try:
    nlp = spacy.load('pt_core_news_sm')
    print('✅ Modelo spaCy português carregado com sucesso!')
except:
    try:
        import pt_core_news_sm
        print('✅ Modelo spaCy português disponível via pacote!')
    except:
        print('⚠️ Modelo spaCy português não encontrado. Usando fallback.')
"

echo "✅ Configuração concluída!"
