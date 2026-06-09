# ============================================================
#  pipeline.py
#  Pipeline RAG complet (orchestration)
# ============================================================
#  Ce module orchestre toutes les étapes du RAG :
#    Documents → Chunks → Embeddings → FAISS
#    → Retrieval → Prompt Augmenté → LLM → Réponse
# ============================================================

import os
import json
import time

from config import DOCUMENTS_DIR, INDEX_DIR, CHUNKS_PATH, TOP_K
from document_loader import DocumentLoader
from embedding_encoder import EmbeddingEncoder
from vector_index import VectorIndex
from retriever import Retriever
from prompt_builder import PromptBuilder
from generator import RAGGenerator


class RAGPipeline:
    """
    Orchestre le pipeline RAG complet de bout en bout.

    C'est la classe principale qui connecte toutes les étapes :
      Documents → Chunks → Embeddings → FAISS → Retrieval
      → Prompt Augmenté → LLM → Réponse
    """

    def __init__(self):
        """Initialise les composants (chargés via initialiser())."""
        self.loader = None
        self.encoder = None
        self.index = None
        self.retriever = None
        self.prompt_builder = None
        self.generator = None
        self.chunks = []
        self.est_initialise = False

    def initialiser(self, chemin_documents=DOCUMENTS_DIR):
        """
        Initialise le pipeline complet :
          1. Charger et découper les documents
          2. Encoder les chunks en embeddings
          3. Construire l'index FAISS
          4. Charger le LLM de génération

        Args:
            chemin_documents: Chemin vers le dossier de documents
        """
        print("\n" + "█" * 60)
        print("█  INITIALISATION DU PIPELINE RAG")
        print("█" * 60 + "\n")

        debut = time.time()

        # Étape 1 : Charger et découper
        self.loader = DocumentLoader()
        self.chunks = self.loader.preparer_corpus(chemin_documents)
        textes = [chunk["texte"] for chunk in self.chunks]

        # Étape 2 : Encoder en embeddings
        self.encoder = EmbeddingEncoder()
        embeddings = self.encoder.encoder_documents(textes)

        # Étape 3 : Construire l'index FAISS
        print("=" * 60)
        print("  📦 Étape 3 : Construction de l'index FAISS")
        print("=" * 60)
        self.index = VectorIndex(dimension=embeddings.shape[1])
        self.index.ajouter(embeddings)
        self.index.sauvegarder()
        self._sauvegarder_chunks()
        print()

        # Étape 4 : Créer les composants
        self.retriever = Retriever(self.encoder, self.index, self.chunks)
        self.prompt_builder = PromptBuilder()

        # Étape 5 : Charger le LLM
        self.generator = RAGGenerator()

        duree = time.time() - debut
        self.est_initialise = True

        print("=" * 60)
        print(f"  ✅ Pipeline RAG initialisé en {duree:.1f} secondes")
        print(f"  ✅ {len(self.chunks)} chunks indexés")
        print(f"  ✅ Prêt à répondre aux questions !")
        print("=" * 60 + "\n")

    def _sauvegarder_chunks(self):
        """Sauvegarde les chunks sur le disque."""
        os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"  💾 Chunks sauvegardés : {CHUNKS_PATH}")

    def poser_question(self, question, top_k=TOP_K, afficher=True):
        """
        Pose une question au système RAG.

        Args:
            question: La question en langage naturel
            top_k: Nombre de documents à récupérer
            afficher: Si True, affiche les résultats détaillés

        Returns:
            dict: {question, reponse, documents, prompt, temps}
        """
        if not self.est_initialise:
            print("  ⚠ Pipeline non initialisé ! Appelez initialiser().")
            return None

        debut = time.time()

        # 1. Rechercher les documents pertinents
        documents = self.retriever.rechercher(question, top_k=top_k)

        # 2. Construire le prompt augmenté
        prompt = self.prompt_builder.construire_prompt(question, documents)

        # 3. Générer la réponse
        reponse = self.generator.generer(prompt)

        duree = time.time() - debut

        resultat = {
            "question": question,
            "reponse": reponse,
            "documents": documents,
            "prompt": prompt,
            "temps": duree,
        }

        if afficher:
            self._afficher_resultat(resultat)

        return resultat

    def poser_question_simple(self, question, afficher=True):
        """
        Pose la même question SANS RAG (pour comparaison).

        Args:
            question: La question en langage naturel
            afficher: Si True, affiche les résultats

        Returns:
            dict: {question, reponse, temps}
        """
        debut = time.time()
        prompt = self.prompt_builder.construire_prompt_simple(question)
        reponse = self.generator.generer(prompt)
        duree = time.time() - debut

        resultat = {
            "question": question,
            "reponse": reponse,
            "temps": duree,
        }

        if afficher:
            print(f"  🤖 Réponse SANS RAG : {reponse}")
            print(f"  ⏱  Temps : {duree:.2f}s")

        return resultat

    def _afficher_resultat(self, resultat):
        """Affiche un résultat RAG de manière détaillée."""
        print("-" * 60)
        print(f"  ❓ Question : {resultat['question']}")
        print("-" * 60)

        print(f"\n  📄 Documents récupérés ({len(resultat['documents'])}) :")
        for i, doc in enumerate(resultat["documents"], 1):
            source = doc.get("source", "?")
            score = doc.get("score", 0)
            extrait = doc["texte"][:120].replace("\n", " ") + "..."
            print(f"    {i}. [{source}] (score: {score:.3f})")
            print(f"       \"{extrait}\"")

        print(f"\n  🤖 Réponse RAG : {resultat['reponse']}")
        print(f"  ⏱  Temps : {resultat['temps']:.2f}s")
        print()
