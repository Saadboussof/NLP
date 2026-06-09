# ============================================================
#  evaluator.py
#  Évaluation et comparaison RAG vs Sans-RAG
# ============================================================
#  Ce module s'occupe de :
#    • Comparer les réponses avec et sans RAG
#    • Évaluer sur plusieurs questions
#    • Afficher un résumé des performances
# ============================================================

import numpy as np


class RAGEvaluator:
    """
    Évalue et compare les réponses avec et sans RAG.

    Permet de démontrer concrètement l'avantage du RAG :
      - La réponse avec RAG est basée sur des faits (nos documents)
      - La réponse sans RAG est une "hallucination" du LLM
    """

    def __init__(self, pipeline):
        """
        Args:
            pipeline: Instance de RAGPipeline initialisée
        """
        self.pipeline = pipeline

    def comparer(self, question):
        """
        Compare la réponse RAG vs sans RAG pour une question.

        Args:
            question: La question à comparer

        Returns:
            dict: Résultats de la comparaison
        """
        print("\n" + "=" * 60)
        print("  ⚖ COMPARAISON : Avec RAG vs Sans RAG")
        print("=" * 60)
        print(f"  ❓ Question : {question}\n")

        # Réponse AVEC RAG
        print("  ── Avec RAG (contexte des documents) ──")
        resultat_rag = self.pipeline.poser_question(question, afficher=False)
        print(f"  🟢 Réponse RAG  : {resultat_rag['reponse']}")
        print(f"  ⏱  Temps        : {resultat_rag['temps']:.2f}s")
        sources = set(d.get("source", "?") for d in resultat_rag["documents"])
        print(f"  📄 Sources      : {', '.join(sources)}")
        print()

        # Réponse SANS RAG
        print("  ── Sans RAG (connaissances internes du LLM) ──")
        resultat_simple = self.pipeline.poser_question_simple(
            question, afficher=False
        )
        print(f"  🔴 Réponse LLM  : {resultat_simple['reponse']}")
        print(f"  ⏱  Temps        : {resultat_simple['temps']:.2f}s")
        print()

        print("  ── Analyse ──")
        print("  → La réponse RAG est basée sur nos documents (vérifiable)")
        print("  → La réponse sans RAG vient des connaissances internes du LLM")
        print("    (peut contenir des hallucinations)")
        print("=" * 60 + "\n")

        return {
            "question": question,
            "reponse_rag": resultat_rag["reponse"],
            "reponse_simple": resultat_simple["reponse"],
            "temps_rag": resultat_rag["temps"],
            "temps_simple": resultat_simple["temps"],
            "nb_sources": len(resultat_rag["documents"]),
        }

    def evaluer_multiple(self, questions):
        """
        Évalue le système sur plusieurs questions.

        Args:
            questions: Liste de questions à tester

        Returns:
            list[dict]: Résultats pour chaque question
        """
        print("\n" + "█" * 60)
        print("█  ÉVALUATION COMPLÈTE DU SYSTÈME RAG")
        print("█" * 60 + "\n")

        resultats = []
        for i, question in enumerate(questions, 1):
            print(f"  ────── Question {i}/{len(questions)} ──────")
            resultat = self.comparer(question)
            resultats.append(resultat)

        # Résumé
        print("\n" + "=" * 60)
        print("  📊 RÉSUMÉ DE L'ÉVALUATION")
        print("=" * 60)
        print(f"  • Questions testées : {len(resultats)}")
        temps_moyen_rag = np.mean([r["temps_rag"] for r in resultats])
        temps_moyen_simple = np.mean([r["temps_simple"] for r in resultats])
        print(f"  • Temps moyen RAG   : {temps_moyen_rag:.2f}s")
        print(f"  • Temps moyen simple: {temps_moyen_simple:.2f}s")
        print("=" * 60 + "\n")

        return resultats
