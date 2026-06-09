# ============================================================
#  2_classification_texte.py
#  Partie 1 - Étape 2 : Classification de texte
# ============================================================
#  Ce script montre comment :
#    • Classifier du texte en catégories prédéfinies
#    • Utiliser un modèle pré-entraîné pour la classification
#    • Classifier en zero-shot (sans entraînement spécifique)
#    • Fine-tuner un modèle pour une tâche personnalisée
# ============================================================

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset


# ============================================================
# SECTION 1 : Classification avec Pipeline (méthode simple)
# ============================================================

def classification_pipeline():
    """
    Utilise la pipeline HuggingFace pour classifier du texte.

    Le modèle par défaut est "distilbert-base-uncased-finetuned-sst-2-english"
    qui classifie en POSITIVE / NEGATIVE (entraîné sur SST-2).
    """
    print("=" * 60)
    print("  Classification de texte avec Pipeline")
    print("=" * 60)

    classifieur = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Textes à classifier
    textes = [
        "This product is absolutely wonderful and exceeded my expectations!",
        "Terrible quality, broke after two days of use.",
        "The delivery was on time and the packaging was decent.",
        "I'm extremely disappointed with the customer service.",
        "Best purchase I've made this year, highly recommended!",
    ]

    print("\n  Résultats de la classification :\n")
    for texte in textes:
        resultat = classifieur(texte)[0]
        label = resultat["label"]
        score = resultat["score"]
        emoji = "✅" if label == "POSITIVE" else "❌"
        barre = "█" * int(score * 30)
        print(f"  {emoji} [{label:8s}] {score:.4f} |{barre}")
        print(f"     \"{texte[:60]}...\"" if len(texte) > 60 else f"     \"{texte}\"")
        print()


# ============================================================
# SECTION 2 : Classification Zero-Shot
# ============================================================

def classification_zero_shot():
    """
    Classification Zero-Shot : classer du texte dans des catégories
    SANS avoir entraîné le modèle sur ces catégories.

    Comment ça marche ?
    Le modèle reformule chaque catégorie comme une hypothèse
    et calcule la probabilité que le texte implique cette hypothèse.

    Exemple : texte = "J'ai mal à la tête"
              catégorie "santé" → hypothèse "Ce texte parle de santé"
              Le modèle calcule P(texte implique hypothèse) = 0.95
    """
    print("=" * 60)
    print("  Classification Zero-Shot")
    print("=" * 60)

    classifieur = pipeline(
        "zero-shot-classification", # this field is used to specify the task name  
        model="facebook/bart-large-mnli"
    )

    # Catégories personnalisées
    categories = ["sports", "politique", "technologie", "santé", "éducation"]

    # Textes à classifier
    textes = [
        "The new iPhone 16 features an improved AI chip and better camera system.",
        "The football world cup final attracted millions of viewers worldwide.",
        "The government announced new policies to reduce carbon emissions.",
        "Regular exercise and a balanced diet can prevent heart disease.",
        "Universities are adopting online learning platforms for remote education.",
    ]

    print(f"\n  Catégories : {categories}\n")

    for texte in textes:
        resultat = classifieur(texte, categories)
        top_label = resultat["labels"][0]
        top_score = resultat["scores"][0]

        print(f"  📄 \"{texte[:65]}...\"" if len(texte) > 65 else f"  📄 \"{texte}\"")
        print(f"     → Catégorie : {top_label} (confiance : {top_score:.4f})")

        # Afficher toutes les scores
        for label, score in zip(resultat["labels"], resultat["scores"]):
            barre = "█" * int(score * 25)
            print(f"       {label:15s} : {score:.4f} |{barre}")
        print()


# ============================================================
# SECTION 3 : Classification manuelle (sans pipeline)
# ============================================================

def classification_manuelle():
    """
    Montre comment faire la classification MANUELLEMENT
    (sans pipeline) pour comprendre ce qui se passe "sous le capot".

    Étapes :
      1. Tokenizer le texte
      2. Passer les tokens dans le modèle
      3. Appliquer softmax pour obtenir les probabilités
      4. Trouver la classe avec la plus haute probabilité
    """
    print("=" * 60)
    print("  Classification manuelle (sans pipeline)")
    print("=" * 60)

    nom_modele = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(nom_modele)
    model = AutoModelForSequenceClassification.from_pretrained(nom_modele)

    # Mapping des labels
    id_to_label = model.config.id2label  # {0: "NEGATIVE", 1: "POSITIVE"}

    texte = "This is a fantastic tutorial on NLP with Transformers!"

    print(f"\n  Texte : \"{texte}\"\n")

    # Étape 1 : Tokenization
    inputs = tokenizer(
        texte,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    print(f"  Étape 1 - Tokenization :")
    print(f"    input_ids shape : {inputs['input_ids'].shape}")
    print(f"    tokens : {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    print()

    # Étape 2 : Passer dans le modèle
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    print(f"  Étape 2 - Sortie du modèle (logits) :")
    print(f"    logits : {logits}")
    print()

    # Étape 3 : Appliquer softmax
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    print(f"  Étape 3 - Probabilités (après softmax) :")
    for i, prob in enumerate(probabilities[0]):
        print(f"    {id_to_label[i]:10s} : {prob:.4f} ({prob*100:.1f}%)")
    print()

    # Étape 4 : Prédiction finale
    prediction = torch.argmax(probabilities, dim=-1).item()
    print(f"  Étape 4 - Prédiction finale :")
    print(f"    Classe : {id_to_label[prediction]}")
    print(f"    Confiance : {probabilities[0][prediction]:.4f}")
    print()


# ============================================================
# SECTION 4 : Classification multi-label
# ============================================================

def classification_multi_label():
    """
    Classification multi-label : un texte peut appartenir
    à PLUSIEURS catégories en même temps.

    Exemple : un article sur "l'IA dans le sport"
    → catégorie "technologie" ET "sport"
    """
    print("=" * 60)
    print("  Classification Multi-Label (Zero-Shot)")
    print("=" * 60)

    classifieur = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    categories = ["science", "business", "health", "entertainment", "politics"]

    textes = [
        "A new AI-powered drug discovery platform has raised $500 million in funding.",
        "The president attended the tech summit to discuss cybersecurity regulations.",
        "Scientists found that video games can improve cognitive function in elderly.",
    ]

    print(f"\n  Catégories : {categories}\n")

    for texte in textes:
        # multi_label=True permet d'attribuer plusieurs catégories
        resultat = classifieur(texte, categories, multi_label=True)

        print(f"  📄 \"{texte}\"")
        print(f"     Catégories attribuées (seuil > 0.3) :")
        for label, score in zip(resultat["labels"], resultat["scores"]):
            if score > 0.3:
                barre = "█" * int(score * 25)
                print(f"       ✓ {label:15s} : {score:.4f} |{barre}")
        print()


# ============================================================
# SECTION 5 : Dataset personnalisé pour le fine-tuning
# ============================================================

class TexteDataset(Dataset):
    """
    Dataset personnalisé pour le fine-tuning d'un modèle
    de classification de texte.

    Ce dataset simule des données d'entraînement avec des
    textes et leurs labels correspondants.
    """

    def __init__(self, textes, labels, tokenizer, max_length=128):
        """
        Args:
            textes: Liste de textes
            labels: Liste de labels (0 ou 1)
            tokenizer: Tokenizer HuggingFace
            max_length: Longueur maximale des séquences
        """
        self.encodings = tokenizer(
            textes,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def demo_fine_tuning():
    """
    Montre la STRUCTURE d'un fine-tuning de modèle
    (avec un petit dataset de démonstration).

    Le fine-tuning = prendre un modèle pré-entraîné et
    l'adapter à NOTRE tâche spécifique avec NOS données.
    """
    print("=" * 60)
    print("  Démonstration de Fine-Tuning (structure)")
    print("=" * 60)

    # Données d'entraînement (petit échantillon de démonstration)
    textes_train = [
        "I absolutely love this product!",
        "Worst purchase ever, total waste of money.",
        "Amazing quality and fast shipping!",
        "The item arrived broken and damaged.",
        "Exceeded all my expectations, will buy again.",
        "Terrible experience, do not recommend.",
        "Great value for the price, very satisfied.",
        "Poor quality, fell apart after one use.",
    ]
    labels_train = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=positif, 0=négatif

    textes_test = [
        "Really happy with my purchase!",
        "Not what I expected, very disappointing.",
    ]
    labels_test = [1, 0]

    # Charger le modèle et le tokenizer
    nom_modele = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(nom_modele)
    model = AutoModelForSequenceClassification.from_pretrained(
        nom_modele,
        num_labels=2,  # 2 classes : positif / négatif
    )

    # Créer les datasets
    train_dataset = TexteDataset(textes_train, labels_train, tokenizer)
    test_dataset = TexteDataset(textes_test, labels_test, tokenizer)

    print(f"\n  Configuration du fine-tuning :")
    print(f"    Modèle de base    : {nom_modele}")
    print(f"    Nombre de classes  : 2 (positif / négatif)")
    print(f"    Données train      : {len(textes_train)} exemples")
    print(f"    Données test       : {len(textes_test)} exemples")
    print()

    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir="./resultats_classification",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        no_cuda=True,  # CPU uniquement pour la démo
    )

    # Créer le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("  ⚠ Structure du fine-tuning configurée.")
    print("    Pour lancer l'entraînement, décommentez : trainer.train()")
    print()

    # Décommenter pour lancer l'entraînement :
    # trainer.train()
    # resultats = trainer.evaluate()
    # print(f"  Résultats : {resultats}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("█  PARTIE 1 - ÉTAPE 2 : CLASSIFICATION DE TEXTE")
    print("█" * 60 + "\n")

    # 1. Classification simple avec pipeline
    print("\n" + "─" * 60)
    print("  📊 SECTION 1 : Classification avec Pipeline")
    print("─" * 60 + "\n")
    classification_pipeline()

    # 2. Classification zero-shot
    print("\n" + "─" * 60)
    print("  🎯 SECTION 2 : Classification Zero-Shot")
    print("─" * 60 + "\n")
    classification_zero_shot()

    # 3. Classification manuelle
    print("\n" + "─" * 60)
    print("  🔧 SECTION 3 : Classification Manuelle")
    print("─" * 60 + "\n")
    classification_manuelle()

    # 4. Multi-label
    print("\n" + "─" * 60)
    print("  🏷️ SECTION 4 : Classification Multi-Label")
    print("─" * 60 + "\n")
    classification_multi_label()

    # 5. Fine-tuning
    print("\n" + "─" * 60)
    print("  🎓 SECTION 5 : Fine-Tuning (structure)")
    print("─" * 60 + "\n")
    demo_fine_tuning()

    print("\n" + "█" * 60)
    print("█  FIN DE LA CLASSIFICATION DE TEXTE")
    print("█" * 60 + "\n")
