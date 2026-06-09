# ============================================================
#  retriever.py
#  Recherche et récupération de documents pertinents
# ============================================================
#  Ce module s'occupe de :
#    • Combiner l'encodeur et l'index FAISS
#    • Trouver les chunks les plus pertinents pour une question
# ============================================================

from config import TOP_K


class Retriever:
    """
    Combine l'encodeur d'embeddings et l'index FAISS pour
    récupérer les documents les plus pertinents pour une requête.

    C'est le "R" dans RAG (Retrieval) — le bibliothécaire
    qui trouve les bons paragraphes en un clin d'œil.
    """

    def __init__(self, encoder, index, chunks):
        """
        Args:
            encoder: Instance d'EmbeddingEncoder
            index: Instance de VectorIndex
            chunks: Liste des chunks (list[dict])
        """
        self.encoder = encoder
        self.index = index
        self.chunks = chunks

    def rechercher(self, requete, top_k=TOP_K):
        """
        Recherche les chunks les plus pertinents pour une question.

        Args:
            requete: La question de l'utilisateur (str)
            top_k: Nombre de résultats à retourner

        Returns:
            list[dict]: Chunks pertinents enrichis avec 'score'
        """
        # 1. Encoder la requête en vecteur
        requete_embedding = self.encoder.encoder_requete(requete)

        # 2. Rechercher dans l'index FAISS
        distances, indices = self.index.rechercher(requete_embedding, top_k)

        # 3. Récupérer les chunks correspondants
        resultats = []
        for dist, idx in zip(distances, indices):
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                # Convertir la distance L2 en score (0 à 1)
                chunk["score"] = float(1 / (1 + dist))
                chunk["distance"] = float(dist)
                resultats.append(chunk)

        return resultats
