# ============================================================
#  embedding_encoder.py
#  Encodage des textes en vecteurs d'embeddings
# ============================================================
#  Ce module s'occupe de :
#    • Charger le modèle sentence-transformers
#    • Encoder des textes (documents ou requêtes) en vecteurs
# ============================================================

import numpy as np
from config import EMBEDDING_MODEL


class EmbeddingEncoder:
    """
    Encode les textes en vecteurs d'embeddings à l'aide de
    sentence-transformers.

    Modèle utilisé : all-MiniLM-L6-v2
      - Dimension : 384
      - Taille : ~80 Mo
      - Rapide et léger, idéal pour un projet étudiant
      - Supporte le français (modèle multilingue)
    """

    def __init__(self, nom_modele=EMBEDDING_MODEL):
        """
        Args:
            nom_modele: Nom du modèle sentence-transformers
        """
        print("=" * 60)
        print("  🧠 Étape 2 : Chargement du modèle d'embeddings")
        print("=" * 60)

        from sentence_transformers import SentenceTransformer
        self.modele = SentenceTransformer(nom_modele)
        self.dimension = self.modele.get_sentence_embedding_dimension()

        print(f"  ✓ Modèle chargé : {nom_modele}")
        print(f"  ✓ Dimension des vecteurs : {self.dimension}")
        print()

    def encoder_documents(self, textes, batch_size=32):
        """
        Encode une liste de textes en vecteurs d'embeddings.

        Args:
            textes: Liste de chaînes de caractères
            batch_size: Nombre de textes à traiter simultanément

        Returns:
            np.ndarray: Matrice (n_textes, dimension)
        """
        print(f"  📊 Encodage de {len(textes)} textes en embeddings...")
        embeddings = self.modele.encode(
            textes,
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        print(f"  ✓ Embeddings créés : shape = {embeddings.shape}")
        print()
        return embeddings.astype("float32")

    def encoder_requete(self, requete):
        """
        Encode une seule requête (question) en vecteur.

        Args:
            requete: La question de l'utilisateur (str)

        Returns:
            np.ndarray: Vecteur (1, dimension)
        """
        embedding = self.modele.encode([requete], convert_to_numpy=True)
        return embedding.astype("float32")
