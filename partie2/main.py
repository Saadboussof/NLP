# ============================================================
#  main.py
#  Point d'entrée — Démonstration du système RAG
# ============================================================
#  Lancer avec : python main.py
# ============================================================

from pipeline import RAGPipeline
from evaluator import RAGEvaluator


def main():
    print("\n" + "█" * 60)
    print("█  PARTIE 2 : SYSTÈME RAG (Retrieval-Augmented Generation)")
    print("█  Mini-Projet NLP — ENSA Al Hoceima — ID2")
    print("█" * 60 + "\n")

    # ── 1. Initialiser le pipeline RAG ──
    rag = RAGPipeline()
    rag.initialiser()

    # ── 2. Poser des questions au système ──
    print("\n" + "█" * 60)
    print("█  DÉMONSTRATION : Questions au système RAG")
    print("█" * 60 + "\n")

    questions_demo = [
        "Qu'est-ce que le mécanisme d'attention dans les Transformers ?",
        "Quelle est la différence entre BERT et GPT ?",
        "Qu'est-ce que la rétropropagation ?",
        "À quoi sert FAISS dans un système RAG ?",
        "Qu'est-ce que l'analyse de sentiment en NLP ?",
    ]

    for question in questions_demo:
        rag.poser_question(question)
        print()

    # ── 3. Comparaison RAG vs Sans-RAG ──
    evaluateur = RAGEvaluator(rag)

    questions_comparaison = [
        "Qu'est-ce que le fine-tuning ?",
        "Comment fonctionne un réseau de neurones récurrent (RNN) ?",
        "Quelles sont les applications de l'intelligence artificielle ?",
    ]

    evaluateur.evaluer_multiple(questions_comparaison)

    print("\n" + "█" * 60)
    print("█  FIN DE LA DÉMONSTRATION DU SYSTÈME RAG")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
