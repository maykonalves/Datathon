# Decision - Co-Piloto de Recrutamento

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.46.1-red.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

O **Decision** é uma aplicação inteligente de recrutamento que utiliza Machine Learning para prever a compatibilidade entre candidatos e vagas. O sistema usa um **modelo semântico unificado**, baseado em *embeddings* de texto (SBERT) e um algoritmo de Gradient Boosting (XGBoost), para analisar o contexto completo de currículos e descrições de vagas, gerando um score de afinidade preciso.

Desenvolvido como solução para o Datathon Data Analytics da Decision, esta ferramenta de IA é focada em duas frentes: a **priorização estratégica de candidatos** para vagas abertas e a **descoberta de perfis de sucesso** com base em dados históricos, visando tornar o processo de seleção mais rápido, assertivo e data-driven.

## Funcionalidades

- **Matching Semântico**: Análise de compatibilidade baseada no significado e contexto dos textos, e não apenas em palavras-chave.
- **Modelo de ML Unificado**: Um único e robusto modelo XGBoost que aprende com embeddings de texto e features de engenharia.
- **Análise de Requisitos**: Avaliação de compatibilidade de níveis de senioridade, idiomas e localização.
- **Interface Streamlit**: Interface web intuitiva para análise de vagas e candidatos.
- **Visualização de Dados**: Gráficos para análise de importância de features e performance do modelo.
- **Containerização Docker**: Deploy simplificado e reproduzível.

## Arquitetura do Projeto

```
Datathon/
├── app/                        # Aplicação Streamlit
│   ├── main.py                 # Ponto de entrada principal
│   └── pages/
│       ├── match.py           # Página de matching
│       └── sobre.py           # Página sobre o projeto
├── data/                      # Dados do projeto
│   ├── raw/                   # Dados brutos
│   │   ├── applicants.json    # Dados dos candidatos
│   │   ├── jobs.json          # Dados das vagas
│   │   └── prospects.json     # Dados de prospects
│   └── processed/             # Dados processados
├── models/                    # Modelos de Machine Learning
│   ├── modelo_treinado*.pkl   # Modelos especializados por tecnologia
│   └── modelo.py              # Script de treinamento
├── notebooks/                 # Análises exploratórias
│   ├── eda_applicants.ipynb   # EDA dos candidatos
│   ├── eda_jobs.ipynb         # EDA das vagas
│   └── eda_prospects.ipynb    # EDA dos prospects
├── Dockerfile                 # Configuração Docker
├── requirements.txt           # Dependências Python
└── README.md                  # Este arquivo
```

## Arquitetura do Modelo

O sistema abandonou a abordagem de múltiplos modelos por tecnologia em favor de um **modelo semântico unificado**, que se mostrou mais flexível e poderoso. A arquitetura funciona da seguinte forma:

1.  **Engenharia de Features**: Dados como nível de senioridade, idiomas e localização são extraídos e comparados.
2.  **Embeddings de Texto**: Os textos completos da vaga e do currículo são convertidos em vetores numéricos (embeddings) usando o modelo **SBERT (`paraphrase-MiniLM-L6-v2`)**. Isso permite que o modelo entenda o significado semântico dos textos.
3.  **Modelo Preditivo**: Os vetores de embeddings e as features de engenharia são combinados para treinar um classificador **XGBoost**, que prevê a probabilidade de um "match" de sucesso.
4.  **Balanceamento de Classes**: A técnica **SMOTE** é usada durante o treinamento para lidar com o desbalanceamento entre candidatos contratados e não contratados.

Esta abordagem permite que o modelo generalize para qualquer tipo de vaga, incluindo tecnologias novas que não estavam nos dados de treino originais.

## Tecnologias Utilizadas

- **Python 3.12+**
- **Streamlit** - Interface web
- **XGBoost** - Algoritmo de Machine Learning
- **Sentence-Transformers (SBERT)** - Processamento de linguagem natural e embeddings
- **NLTK** - Toolkit de linguagem natural (para stopwords)
- **Imbalanced-learn** - Para balanceamento de classes (SMOTE)
- **Pandas/NumPy** - Manipulação de dados
- **Scikit-learn** - Ferramentas de ML
- **Docker** - Containerização

## Instalação e Configuração

### Pré-requisitos

- Python 3.12 ou superior
- Docker (opcional)
- Git

### Instalação Local

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/maykonalves/Datathon.git](https://github.com/maykonalves/Datathon.git)
    cd Datathon
    ```

2.  **Crie um ambiente virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # ou
    venv\Scripts\activate     # Windows
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: A primeira execução irá baixar os modelos de embedding do Sentence-Transformers, o que pode levar alguns minutos.*

4.  **Execute o treinamento do modelo:**
    ```bash
    python models/modelo.py
    ```

5.  **Execute a aplicação:**
    ```bash
    streamlit run app/main.py
    ```

### Instalação com Docker

1.  **Build da imagem:**
    ```bash
    docker build -t decision-app .
    ```

2.  **Execute o container:**
    ```bash
    docker run -p 8501:8501 decision-app
    ```

A aplicação estará disponível em `http://localhost:8501`.

## Como Usar

### Página de Match

1.  Acesse a aplicação e navegue para "Match Análise".
2.  Preencha as informações da vaga e do candidato nos campos da tela.
3.  Clique em "Calcular Match".
4.  O sistema irá processar os dados, gerar as features e embeddings, e exibir um score de afinidade.

### Interpretação dos Resultados

-   **Score de Afinidade**: Probabilidade de sucesso (0-100%) calculada pelo modelo.
-   **Análise Detalhada**: Um resumo simplificado de requisitos-chave (nível profissional, idioma) para feedback rápido.

## Troubleshooting

### Problemas Comuns

1.  **Erro de modelo não encontrado (`.pkl`):**
    -   Certifique-se de ter executado o script `python models/modelo.py` para treinar e salvar o modelo.

2.  **Problemas de dependências:**
    -   Verifique se está usando Python 3.12+.
    -   Recrie seu ambiente virtual e reinstale as dependências: `pip install -r requirements.txt --force-reinstall`

3.  **Erro no download do modelo de embedding:**
    -   Verifique sua conexão com a internet. A biblioteca `sentence-transformers` precisa baixar modelos do Hugging Face na primeira vez que é usada.

## Licença

Este projeto está sob a licença MIT.
