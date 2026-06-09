# 🤖 Mini-Projet : Transformers et Systèmes RAG

**ENSA Al Hoceima — ID2**

---

## 📋 Description

Ce mini-projet explore l'utilisation pratique des modèles Transformers pour le NLP (Natural Language Processing) et l'implémentation d'un système RAG (Retrieval-Augmented Generation).

## 📁 Structure du Projet

```
NLP/
├── partie1/                              # ✅ IMPLÉMENTÉE
│   ├── __init__.py
│   ├── 1_exploration_transformers.py     # Chargement modèles, tokenizers, pipelines
│   ├── 2_classification_texte.py         # Classification de texte
│   ├── 3_analyse_sentiment.py            # Analyse de sentiment
│   ├── 4_question_answering.py           # Question Answering
│   └── 5_resume_automatique.py           # Résumé automatique
│
├── partie2/                              # ✅ IMPLÉMENTÉE
│   ├── __init__.py
│   ├── config.py                         # Configuration et paramètres
│   ├── document_loader.py                # Chargement et chunking
│   ├── embedding_encoder.py              # Modèle d'embeddings
│   ├── vector_index.py                   # Indexation FAISS
│   ├── retriever.py                      # Recherche vectorielle
│   ├── prompt_builder.py                 # Création des prompts
│   ├── generator.py                      # Génération avec LLM local
│   ├── pipeline.py                       # Orchestration RAG
│   ├── evaluator.py                      # Évaluation et comparaison
│   ├── main.py                           # Point d'entrée principal
│   └── documents/                        # Corpus de test (.txt)
│
├── rapport/
│   └── rapport_partie1.md                # Rapport détaillé de la Partie 1
│
├── requirements.txt                      # Dépendances globales
└── README.md                             # Ce fichier
```

## 🚀 Installation

```bash
# 1. Cloner ou télécharger le projet
# 2. Installer les dépendances
pip install -r requirements.txt
```

## ▶️ Exécution

```bash
# Exécuter chaque script individuellement :
python partie1/1_exploration_transformers.py
python partie1/2_classification_texte.py
python partie1/3_analyse_sentiment.py
python partie1/4_question_answering.py
python partie1/5_resume_automatique.py
```

## 📝 Partie 1 : Exploration des Transformers

| Script | Tâche | Modèles Utilisés |
|--------|-------|-------------------|
| `1_exploration_transformers.py` | Modèles, Tokenizers, Pipelines | BERT, GPT-2, DistilBERT |
| `2_classification_texte.py` | Classification de texte | DistilBERT, BART-MNLI |
| `3_analyse_sentiment.py` | Analyse de sentiment | DistilBERT-SST2, RoBERTa |
| `4_question_answering.py` | Question Answering | DistilBERT-SQuAD, XLM-R |
| `5_resume_automatique.py` | Résumé automatique | BART-CNN, T5, DistilBART |

## 📝 Partie 2 : Système RAG

L'implémentation complète et modulaire se trouve dans `partie2/` :
- `main.py` : Lancez ce script pour voir la démo complète
- Le système charge des documents texte locaux (`documents/`)
- Utilise `sentence-transformers` et `FAISS` pour la recherche vectorielle
- Utilise `Flan-T5` (modèle local) pour générer les réponses augmentées
- Compare automatiquement les réponses avec et sans RAG
