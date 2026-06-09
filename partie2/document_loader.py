# ============================================================
#  document_loader.py
#  Chargement et préparation des documents
# ============================================================
#  Ce module s'occupe de :
#    • Charger des fichiers texte (.txt) et PDF (.pdf)
#    • Découper les textes en morceaux (chunks) avec chevauchement
#    • Préparer le corpus pour l'indexation
# ============================================================

import os
import glob
from config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentLoader:
    """
    Charge des documents depuis différentes sources et les
    découpe en morceaux (chunks) pour l'indexation.

    Pourquoi découper en chunks ?
      - Les modèles d'embeddings ont une limite de tokens (~256-512)
      - Des chunks plus petits = une recherche plus précise
      - Le chevauchement évite de couper une idée en deux
    """

    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        """
        Args:
            chunk_size: Taille de chaque morceau en caractères
            chunk_overlap: Chevauchement entre les morceaux
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def charger_texte(self, chemin):
        """
        Charge un fichier texte (.txt).

        Args:
            chemin: Chemin vers le fichier texte

        Returns:
            str: Contenu du fichier
        """
        with open(chemin, "r", encoding="utf-8") as f:
            contenu = f.read()
        return contenu

    def charger_pdf(self, chemin):
        """
        Charge un fichier PDF et extrait le texte.

        Args:
            chemin: Chemin vers le fichier PDF

        Returns:
            str: Texte extrait du PDF
        """
        try:
            from pypdf import PdfReader
            reader = PdfReader(chemin)
            texte = ""
            for page in reader.pages:
                texte += page.extract_text() + "\n"
            return texte
        except ImportError:
            print("  ⚠ pypdf n'est pas installé. Installez-le : pip install pypdf")
            return ""

    def charger_dossier(self, chemin_dossier):
        """
        Charge tous les fichiers .txt et .pdf d'un dossier.

        Args:
            chemin_dossier: Chemin vers le dossier de documents

        Returns:
            list[dict]: Liste de dicts {'nom': str, 'contenu': str}
        """
        documents = []

        # Charger les fichiers .txt
        for fichier in sorted(glob.glob(os.path.join(chemin_dossier, "*.txt"))):
            nom = os.path.basename(fichier)
            contenu = self.charger_texte(fichier)
            if contenu.strip():
                documents.append({"nom": nom, "contenu": contenu})
                print(f"  ✓ Chargé : {nom} ({len(contenu):,} caractères)")

        # Charger les fichiers .pdf
        for fichier in sorted(glob.glob(os.path.join(chemin_dossier, "*.pdf"))):
            nom = os.path.basename(fichier)
            contenu = self.charger_pdf(fichier)
            if contenu.strip():
                documents.append({"nom": nom, "contenu": contenu})
                print(f"  ✓ Chargé : {nom} ({len(contenu):,} caractères)")

        return documents

    def decouper_en_chunks(self, texte, source=""):
        """
        Découpe un texte en morceaux (chunks) avec chevauchement.

        Exemple avec chunk_size=10 et chunk_overlap=3 :
          Texte : "ABCDEFGHIJKLMNOPQR"
          Chunk 1 : "ABCDEFGHIJ"     (positions 0-9)
          Chunk 2 : "HIJKLMNOPQ"     (positions 7-16, chevauche de 3)
          Chunk 3 : "OPQR"           (positions 14-17)

        Args:
            texte: Le texte à découper
            source: Nom du document source (pour traçabilité)

        Returns:
            list[dict]: Liste de chunks {'texte', 'source', 'index'}
        """
        chunks = []
        debut = 0
        index = 0

        while debut < len(texte):
            fin = debut + self.chunk_size
            morceau = texte[debut:fin]

            # Couper à la fin d'une phrase si possible
            if fin < len(texte):
                dernier_point = morceau.rfind(".")
                if dernier_point > self.chunk_size // 2:
                    morceau = morceau[: dernier_point + 1]
                    fin = debut + dernier_point + 1

            morceau = morceau.strip()

            if morceau:
                chunks.append({
                    "texte": morceau,
                    "source": source,
                    "index": index,
                })
                index += 1

            debut = fin - self.chunk_overlap

        return chunks

    def preparer_corpus(self, chemin_dossier):
        """
        Pipeline complet : charger les documents et les découper.

        Args:
            chemin_dossier: Chemin vers le dossier de documents

        Returns:
            list[dict]: Tous les chunks de tous les documents
        """
        print("=" * 60)
        print("  📂 Étape 1 : Chargement des documents")
        print("=" * 60)

        documents = self.charger_dossier(chemin_dossier)
        print(f"\n  → {len(documents)} document(s) chargé(s)\n")

        print("  ✂ Découpage en chunks...")
        tous_les_chunks = []
        for doc in documents:
            chunks = self.decouper_en_chunks(doc["contenu"], source=doc["nom"])
            tous_les_chunks.extend(chunks)
            print(f"    • {doc['nom']} → {len(chunks)} chunks")

        print(f"\n  → Total : {len(tous_les_chunks)} chunks créés")
        print()
        return tous_les_chunks
