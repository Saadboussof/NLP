# ============================================================
#  vector_index.py
#  Indexation vectorielle avec FAISS
# ============================================================
#  Ce module s'occupe de :
#    • Créer un index FAISS pour la recherche par similarité
#    • Ajouter des vecteurs, rechercher les plus proches
#    • Sauvegarder / charger l'index sur le disque
# ============================================================

import os
from config import EMBEDDING_DIMENSION, TOP_K, INDEX_PATH


class VectorIndex:
    """
    Crée et gère un index vectoriel FAISS pour la recherche
    rapide par similarité.

    FAISS (Facebook AI Similarity Search) :
      - Développé par Meta (Facebook)
      - Recherche parmi des millions de vecteurs en millisecondes
      - Utilise des algorithmes de clustering (K-Means)
        pour accélérer la recherche
    """

    def __init__(self, dimension=EMBEDDING_DIMENSION):
        """
        Args:
            dimension: Dimension des vecteurs d'embeddings
        """
        import faiss
        self.dimension = dimension
        # IndexFlatL2 = recherche exhaustive par distance L2
        # C'est le plus simple et le plus précis
        # Pour un grand corpus (>100k), utiliser IndexIVFFlat
        self.index = faiss.IndexFlatL2(dimension)
        self.faiss = faiss

    def ajouter(self, embeddings):
        """
        Ajoute des vecteurs dans l'index FAISS.

        Args:
            embeddings: np.ndarray (n, dimension)
        """
        self.index.add(embeddings)
        print(f"  ✓ {embeddings.shape[0]} vecteurs ajoutés à l'index FAISS")
        print(f"  ✓ Total dans l'index : {self.index.ntotal} vecteurs")

    def rechercher(self, requete_embedding, top_k=TOP_K):
        """
        Recherche les vecteurs les plus proches d'une requête.

        Args:
            requete_embedding: Vecteur de la requête (1, dimension)
            top_k: Nombre de résultats à retourner

        Returns:
            tuple: (distances, indices) — tableaux numpy
        """
        distances, indices = self.index.search(requete_embedding, top_k)
        return distances[0], indices[0]

    def sauvegarder(self, chemin=INDEX_PATH):
        """Sauvegarde l'index FAISS sur le disque."""
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        self.faiss.write_index(self.index, chemin)
        print(f"  💾 Index FAISS sauvegardé : {chemin}")

    def charger(self, chemin=INDEX_PATH):
        """Charge un index FAISS depuis le disque."""
        if os.path.exists(chemin):
            self.index = self.faiss.read_index(chemin)
            print(f"  ✓ Index FAISS chargé : {self.index.ntotal} vecteurs")
        else:
            print(f"  ⚠ Fichier non trouvé : {chemin}")
