# ============================================================
#  prompt_builder.py
#  Construction du prompt augmenté
# ============================================================
#  Ce module s'occupe de :
#    • Formater les documents récupérés en contexte lisible
#    • Construire le prompt augmenté (question + contexte)
#    • Construire un prompt simple (sans RAG, pour comparaison)
# ============================================================


class PromptBuilder:
    """
    Construit le prompt augmenté en combinant la question
    de l'utilisateur avec les documents récupérés par FAISS.

    Le prompt augmenté = le secret de RAG.
    Au lieu d'envoyer juste la question au LLM, on lui donne
    aussi le contexte pertinent extrait de nos documents.
    """

    def formater_contexte(self, documents):
        """
        Formate les documents récupérés en un bloc de contexte.

        Args:
            documents: Liste de chunks avec 'texte', 'source', 'score'

        Returns:
            str: Contexte formaté
        """
        contexte = ""
        for i, doc in enumerate(documents, 1):
            source = doc.get("source", "inconnu")
            score = doc.get("score", 0)
            contexte += f"--- Document {i} (source: {source}, "
            contexte += f"pertinence: {score:.2f}) ---\n"
            contexte += doc["texte"] + "\n\n"
        return contexte.strip()

    def construire_prompt(self, question, documents):
        """
        Construit le prompt augmenté complet.

        Args:
            question: La question de l'utilisateur
            documents: Les documents récupérés par le Retriever

        Returns:
            str: Le prompt augmenté prêt pour le LLM
        """
        contexte = self.formater_contexte(documents)

        prompt = (
            "Utilise les informations suivantes pour répondre à la question. "
            "Si la réponse n'est pas dans le contexte, dis-le clairement.\n\n"
            f"Contexte :\n{contexte}\n\n"
            f"Question : {question}\n\n"
            "Réponse :"
        )
        return prompt

    def construire_prompt_simple(self, question):
        """
        Construit un prompt SANS contexte (pour comparer avec RAG).

        Args:
            question: La question de l'utilisateur

        Returns:
            str: Le prompt simple (sans RAG)
        """
        return f"Réponds à cette question : {question}\n\nRéponse :"
