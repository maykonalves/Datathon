# Co-Piloto de Recrutamento IA - Datathon Decision

## Resumo do Projeto

Este projeto foi desenvolvido como solução para o Datathon Data Analytics da Decision. Trata-se de uma aplicação web, construída com **Streamlit** e conteinerizada com **Docker**, que atua como um "Co-Piloto de Recrutamento". A ferramenta de IA é focada em duas frentes: a **priorização estratégica de candidatos** para vagas abertas e a **descoberta de perfis de sucesso** com base em dados históricos, visando tornar o processo de seleção mais rápido, assertivo e data-driven.

## O Desafio

O processo de recrutamento da Decision, embora robusto, enfrenta desafios de eficiência e escalabilidade. As principais dores identificadas foram:

* **Busca Manual e Demorada:** Encontrar o candidato ideal em diversas plataformas consome um tempo valioso.
* **Falta de Padronização:** A ausência de um padrão em entrevistas pode levar à perda de informações cruciais sobre os candidatos.
* **Dificuldade de Análise:** Avaliar o "match" técnico e o engajamento real de centenas de candidatos para uma vaga é um processo complexo e suscetível a vieses.

## A Solução Proposta

Nossa solução é uma plataforma de IA que empodera os recrutadores com insights e recomendações inteligentes através de duas funcionalidades principais.

### Funcionalidades Principais

#### 1. Matriz de Priorização de Talentos

Esta é a funcionalidade central da aplicação. Em vez de um simples ranking, ela oferece uma análise estratégica bidimensional para cada candidato em relação a uma vaga, utilizando dois modelos preditivos distintos:

* **Score de Compatibilidade Técnica (Modelo 1):** Prevê o quão bem as habilidades e a experiência de um candidato se alinham aos requisitos da vaga. Responde à pergunta: "Este candidato **pode** fazer o trabalho?".
* **Score de Previsão de Aceite (Modelo 2):** Prevê a probabilidade de um candidato aceitar uma proposta caso ela seja oferecida, medindo seu engajamento e interesse. Responde à pergunta: "Este candidato **quer** este trabalho?".

Esses dois scores posicionam cada candidato em uma **Matriz de Talentos 2x2**, segmentando-os em quatro quadrantes estratégicos:
* **Melhores candidatos (Técnico Alto / Aceite Alto):** Prioridade máxima.
* **Talentos em Risco (Técnico Alto / Aceite Baixo):** Requerem uma abordagem consultiva para garantir o aceite.
* **Potenciais a Desenvolver (Técnico Baixo / Aceite Alto):** Ótimos para vagas flexíveis ou com treinamento.
* **Baixa Prioridade (Técnico Baixo / Aceite Baixo):** Devem ser despriorizados para a vaga atual.

A funcionalidade também conta com **IA Explicável (XAI)**, usando a biblioteca `SHAP` para detalhar os fatores que mais influenciaram ambos os scores, garantindo total transparência.

#### 2. Análise de Perfis de Sucesso

Esta funcionalidade utiliza **aprendizagem não-supervisionada (Clustering)** para analisar os dados históricos de todos os candidatos já contratados pela Decision.

* **Descoberta de Personas:** O modelo agrupa os candidatos de sucesso em clusters, revelando "personas" ou perfis ideais que a empresa consegue atrair e contratar com eficiência (ex: "Especialistas SAP Sênior bilíngues", "Desenvolvedores Java Pleno para o setor financeiro").
* **Padronização e Estratégia:** Ajuda a empresa a entender seus nichos de mercado, a padronizar o que constitui um "candidato de sucesso" e a guiar a busca ativa por novos talentos de forma mais estratégica.

---

## Arquitetura do Projeto

O projeto segue uma arquitetura modular para garantir a separação de conceitos entre a lógica da aplicação (backend) e a interface do usuário (frontend).

```text
recruitment_ai_datathon/
│
├── app/                  # Frontend (Streamlit)
│   ├── pages/            # Páginas da aplicação
│   └── main.py           # Página de configuração do streamlit
│
├── data/
│   ├── raw/              # Dados brutos .json
│   └── processed/        # Dados processados e unificados
│
├── models/               # Modelos de ML e pré-processadores salvos (.pkl)
│
├── notebooks/            # Notebooks para Análise Exploratória (EDA)
│
├── src/                  # Backend e lógica de negócio
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── predictor.py
│
├── .gitignore
├── Dockerfile            # Definição do contêiner da aplicação
├── requirements.txt      # Dependências Python
└── README.md