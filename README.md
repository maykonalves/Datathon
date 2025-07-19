# Decision - Co-Piloto de Recrutamento

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.46.1-red.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

## Descrição

O **Decision** é uma aplicação inteligente de recrutamento que utiliza Machine Learning para fazer matching entre candidatos e vagas de emprego. O sistema possui modelos especializados por tecnologia (Python, Java, SQL, AWS, Azure, SAP, React, Oracle) e um modelo geral como fallback, proporcionando alta precisão na recomendação de candidatos.

Desenvolvido como solução para o Datathon Data Analytics da Decision, esta ferramenta de IA é focada em duas frentes: a **priorização estratégica de candidatos** para vagas abertas e a **descoberta de perfis de sucesso** com base em dados históricos, visando tornar o processo de seleção mais rápido, assertivo e data-driven.

## Funcionalidades

- **Matching Inteligente**: Algoritmos de ML especializados por área de tecnologia
- **Análise Semântica**: Processamento de linguagem natural para análise de currículos e descrições de vagas
- **Modelos Especializados**: 8 modelos específicos para diferentes tecnologias
- **Interface Streamlit**: Interface web intuitiva e responsiva
- **SHAP Explainability**: Visualizações de interpretabilidade dos modelos
- **Containerização Docker**: Deploy simplificado

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
│   ├── modelo_*.pkl           # Modelos especializados por tecnologia
│   ├── modelo.py              # Script de treinamento
│   ├── models_manifest.json   # Manifesto dos modelos
│   └── shap_summary_*.png     # Visualizações SHAP
├── notebooks/                 # Análises exploratórias
│   ├── eda_applicants.ipynb   # EDA dos candidatos
│   ├── eda_jobs.ipynb         # EDA das vagas
│   └── eda_prospects.ipynb    # EDA dos prospects
├── src/                       # Código fonte adicional
├── Dockerfile                 # Configuração Docker
├── requirements.txt           # Dependências Python
└── README.md                  # Este arquivo
```

## Modelos Especializados

O sistema conta com os seguintes modelos especializados:

1. **Python** - Para vagas relacionadas a Python, Django, Flask, Data Science
2. **Java** - Para vagas relacionadas a Java, Spring, Hibernate
3. **SQL** - Para vagas relacionadas a bancos de dados e SQL
4. **AWS** - Para vagas relacionadas a Amazon Web Services
5. **Azure** - Para vagas relacionadas a Microsoft Azure
6. **SAP** - Para vagas relacionadas a sistemas SAP
7. **React** - Para vagas relacionadas a React e frontend
8. **Oracle** - Para vagas relacionadas a tecnologias Oracle
9. **General** - Modelo fallback para outras tecnologias

## Tecnologias Utilizadas

- **Python 3.12+**
- **Streamlit** - Interface web
- **XGBoost** - Algoritmo de Machine Learning
- **spaCy** - Processamento de linguagem natural
- **NLTK** - Toolkit de linguagem natural
- **SHAP** - Explicabilidade dos modelos
- **Pandas/NumPy** - Manipulação de dados
- **Scikit-learn** - Ferramentas de ML
- **Docker** - Containerização

## Instalação e Configuração

### Pré-requisitos

- Python 3.12 ou superior
- Docker (opcional)
- Git

### Instalação Local

1. **Clone o repositório:**
```bash
git clone https://github.com/maykonalves/Datathon.git
cd Datathon
```

2. **Crie um ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Baixe o modelo spaCy em português:**
```bash
python -m spacy download pt_core_news_sm
```

5. **Execute o treinamento dos modelos (se necessário):**
```bash
cd models
python modelo.py
```

6. **Execute a aplicação:**
```bash
streamlit run app/main.py
```

### Instalação com Docker

1. **Build da imagem:**
```bash
docker build -t decision-app .
```

2. **Execute o container:**
```bash
docker run -p 8501:8501 decision-app
```

A aplicação estará disponível em `http://localhost:8501`

## Como Usar

### Página de Match

1. Acesse a aplicação no navegador
2. Navegue para "Match perfil"
3. Selecione uma vaga disponível no sistema
4. O sistema irá:
   - Identificar automaticamente a tecnologia principal da vaga
   - Selecionar o modelo especializado apropriado
   - Calcular scores de matching para todos os candidatos
   - Exibir os candidatos ranqueados por compatibilidade

### Interpretação dos Resultados

- **Score de Match**: Probabilidade de sucesso (0-100%)
- **Modelo Utilizado**: Qual modelo especializado foi aplicado
- **Features Principais**: Principais fatores que influenciaram o score

## Desenvolvimento e Contribuição

### Estrutura do Código

- `app/main.py`: Configuração principal do Streamlit
- `app/pages/match.py`: Lógica de matching e interface
- `models/modelo.py`: Script de treinamento dos modelos
- `notebooks/`: Análises exploratórias e desenvolvimento

### Executando os Notebooks

Para executar as análises exploratórias:

```bash
jupyter lab notebooks/
```

### Retreinamento dos Modelos

Para retreinar os modelos com novos dados:

```bash
cd models
python modelo.py
```

## Métricas e Performance

Os modelos são avaliados usando:
- **Accuracy**
- **Precision/Recall**
- **F1-Score**
- **Cross-validation** com StratifiedKFold

Visualizações SHAP são geradas automaticamente para interpretabilidade.

## Deploy

### Dockerfile

O projeto inclui um Dockerfile otimizado para produção:
- Baseado em Python 3.12 slim
- Health check integrado
- Exposição na porta 8501
- Configuração para servidor público

### Variáveis de Ambiente

Nenhuma variável de ambiente específica é necessária para execução básica.

## Troubleshooting

### Problemas Comuns

1. **Modelo spaCy não encontrado:**
   ```bash
   python -m spacy download pt_core_news_sm
   ```

2. **Erro de modelos não encontrados:**
   - Execute `python models/modelo.py` para treinar os modelos

3. **Problemas de dependências:**
   - Verifique se está usando Python 3.12+
   - Reinstale as dependências: `pip install -r requirements.txt --force-reinstall`

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Equipe

Desenvolvido durante o Datathon por **Maykon Alves**.

## Contato

- GitHub: [@maykonalves](https://github.com/maykonalves)
- LinkedIn: [Maykon Alves](https://linkedin.com/in/maykonalves)

---

**Decision** - Transformando recrutamento com Inteligência Artificial