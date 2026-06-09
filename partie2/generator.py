# ============================================================
#  generator.py
#  Génération de réponses avec un LLM local
# ============================================================
#  Ce module s'occupe de :
#    • Charger le modèle de génération (Flan-T5)
#    • Générer des réponses à partir de prompts
# ============================================================

from config import GENERATOR_MODEL, MAX_NEW_TOKENS, TEMPERATURE


class RAGGenerator:
    """
    Génère la réponse finale en utilisant un modèle de langage (LLM).

    Modèle utilisé : google/flan-t5-base
      - Modèle de Google, spécialisé en Q&A
      - Taille : ~250 Mo, fonctionne sur CPU
      - Pas besoin de clé API
    """

    def __init__(self, nom_modele=GENERATOR_MODEL):
        """
        Args:
            nom_modele: Nom du modèle HuggingFace à utiliser
        """
        print("=" * 60)
        print("  🤖 Chargement du modèle de génération (LLM)")
        print("=" * 60)

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.tokenizer = AutoTokenizer.from_pretrained(nom_modele)
        self.modele = AutoModelForSeq2SeqLM.from_pretrained(nom_modele)

        print(f"  ✓ Modèle chargé : {nom_modele}")
        nb_params = sum(p.numel() for p in self.modele.parameters())
        print(f"  ✓ Paramètres : {nb_params:,}")
        print()

    def generer(self, prompt, max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
        """
        Génère une réponse à partir d'un prompt.

        Args:
            prompt: Le prompt (texte d'entrée)
            max_tokens: Nombre max de tokens à générer
            temperature: Créativité (0=déterministe, 1=créatif)

        Returns:
            str: La réponse générée
        """
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        with torch.no_grad():
            outputs = self.modele.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                num_beams=2,
            )

        reponse = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return reponse
