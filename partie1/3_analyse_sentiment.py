# ============================================================
#  3_analyse_sentiment.py
#  Partie 1 - Étape 3 : Analyse de Sentiment
# ============================================================
#  Ce script montre comment :
#    • Détecter le sentiment (positif/négatif/neutre) d'un texte
#    • Utiliser différents modèles spécialisés
#    • Analyser le sentiment sur des avis, tweets, etc.
#    • Visualiser les résultats d'analyse
# ============================================================

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)


# ============================================================
# SECTION 1 : Analyse de sentiment basique
# ============================================================

def analyse_sentiment_basique():
    """
    Analyse de sentiment simple avec la pipeline par défaut.

    Le sentiment est classifié en :
      - POSITIVE (texte positif, content, satisfait)
      - NEGATIVE (texte négatif, mécontent, triste)
    """
    print("=" * 60)
    print("  Analyse de Sentiment - Méthode Basique")
    print("=" * 60)

    analyseur = pipeline("sentiment-analysis")

    # Avis sur des produits / services
    avis = [
        "The food at this restaurant was absolutely delicious!",
        "I waited 3 hours for my order and it was cold when it arrived.",
        "Decent product, works as expected. Nothing special.",
        "This is hands down the best laptop I have ever owned!",
        "The hotel room was dirty and the staff was incredibly rude.",
        "Acceptable quality for the price. Would consider buying again.",
    ]

    print("\n  📝 Analyse d'avis clients :\n")

    resultats = []
    for avis_texte in avis:
        result = analyseur(avis_texte)[0]
        resultats.append(result)

        # Barre visuelle
        score = result["score"]
        if result["label"] == "POSITIVE":
            emoji = "😊"
            couleur_barre = "🟢"
        else:
            emoji = "😞"
            couleur_barre = "🔴"

        barre = "█" * int(score * 30)
        print(f"  {emoji} [{result['label']:8s}] {score:.4f} |{barre}")
        print(f"     \"{avis_texte}\"")
        print()

    # Statistiques
    positifs = sum(1 for r in resultats if r["label"] == "POSITIVE")
    negatifs = len(resultats) - positifs
    print(f"  📊 Statistiques :")
    print(f"     Positifs : {positifs}/{len(resultats)} ({positifs/len(resultats)*100:.0f}%)")
    print(f"     Négatifs : {negatifs}/{len(resultats)} ({negatifs/len(resultats)*100:.0f}%)")
    print()


# ============================================================
# SECTION 2 : Analyse de sentiment multi-classe
# ============================================================

def analyse_sentiment_multiclasse():
    """
    Analyse de sentiment avec 5 classes (étoiles) :
      ⭐ = très négatif
      ⭐⭐ = négatif
      ⭐⭐⭐ = neutre
      ⭐⭐⭐⭐ = positif
      ⭐⭐⭐⭐⭐ = très positif

    Utilise le modèle nlptown/bert-base-multilingual-uncased-sentiment
    qui supporte plusieurs langues (anglais, français, allemand, etc.)
    """
    print("=" * 60)
    print("  Analyse de Sentiment Multi-Classe (1-5 étoiles)")
    print("=" * 60)

    analyseur = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

    # Textes en différentes langues
    textes = [
        ("EN", "This movie is absolutely brilliant, a masterpiece!"),
        ("FR", "Ce film est absolument nul, une perte de temps totale."),
        ("EN", "Average food, nothing to complain about but nothing exciting."),
        ("FR", "Excellent service, je recommande vivement ce restaurant!"),
        ("EN", "Not bad, but could be much better with some improvements."),
        ("FR", "Horrible expérience, je ne reviendrai jamais."),
    ]

    print("\n  📝 Analyse multi-langue :\n")

    for langue, texte in textes:
        result = analyseur(texte)[0]
        # Le label est sous forme "X stars"
        nb_etoiles = int(result["label"][0])
        etoiles = "⭐" * nb_etoiles + "☆" * (5 - nb_etoiles)

        print(f"  [{langue}] {etoiles} ({result['score']:.4f})")
        print(f"       \"{texte}\"")
        print()


# ============================================================
# SECTION 3 : Analyse de sentiment sur des tweets
# ============================================================

def analyse_tweets():
    """
    Analyse de sentiment spécialisée pour les tweets/réseaux sociaux.

    Les tweets ont des particularités :
      - Emojis, abréviations, hashtags
      - Texte court et informel
      - Sarcasme fréquent

    Utilise un modèle entraîné spécifiquement sur des tweets.
    """
    print("=" * 60)
    print("  Analyse de Sentiment - Tweets / Réseaux Sociaux")
    print("=" * 60)

    # Modèle spécialisé pour les tweets (3 classes)
    analyseur = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    tweets = [
        "Just got my new phone and it's amazing! 📱🔥 #tech #upgrade",
        "Stuck in traffic for 2 hours. Worst Monday ever 😤",
        "The weather is nice today, might go for a walk 🌤️",
        "Can't believe they cancelled my favorite show! 😭💔",
        "Had an okay lunch, nothing extraordinary 🍕",
        "So excited for the concert tonight!! 🎵🎶 Let's gooo!",
    ]

    print("\n  🐦 Analyse de tweets :\n")

    for tweet in tweets:
        result = analyseur(tweet)[0]
        label = result["label"]
        score = result["score"]

        # Mapper les labels
        if "positive" in label.lower():
            emoji = "😊"
            sentiment = "POSITIF"
        elif "negative" in label.lower():
            emoji = "😞"
            sentiment = "NÉGATIF"
        else:
            emoji = "😐"
            sentiment = "NEUTRE "

        barre = "█" * int(score * 25)
        print(f"  {emoji} [{sentiment}] {score:.4f} |{barre}")
        print(f"     \"{tweet}\"")
        print()


# ============================================================
# SECTION 4 : Analyse manuelle avec scores détaillés
# ============================================================

def analyse_sentiment_detaillee():
    """
    Analyse manuelle qui montre les scores pour CHAQUE classe.

    Au lieu d'utiliser la pipeline, on fait tout manuellement
    pour voir les probabilités de chaque sentiment.
    """
    print("=" * 60)
    print("  Analyse de Sentiment Détaillée (manuelle)")
    print("=" * 60)

    nom_modele = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(nom_modele)
    model = AutoModelForSequenceClassification.from_pretrained(nom_modele)

    textes = [
        "I can't believe how good this movie was!",
        "The service was absolutely terrible.",
        "It's an average day, nothing special happening.",
    ]

    # Labels du modèle
    labels = ["Négatif", "Neutre", "Positif"]

    print()
    for texte in textes:
        # Tokenization
        inputs = tokenizer(texte, return_tensors="pt", padding=True, truncation=True)

        # Prédiction
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probas = torch.nn.functional.softmax(logits, dim=-1)[0]

        # Trouver la classe dominante
        prediction = torch.argmax(probas).item()

        print(f"  📄 \"{texte}\"")
        print(f"  ╔{'═' * 40}╗")
        for i, (label, proba) in enumerate(zip(labels, probas)):
            barre = "█" * int(proba * 25)
            fleche = " ◄── " if i == prediction else "     "
            print(f"  ║ {label:10s}: {proba:.4f} |{barre:25s}║{fleche}")
        print(f"  ╚{'═' * 40}╝")
        print(f"  → Sentiment dominant : {labels[prediction]} ({probas[prediction]:.4f})")
        print()


# ============================================================
# SECTION 5 : Analyse de sentiment par aspect
# ============================================================

def analyse_sentiment_aspect():
    """
    Analyse de sentiment par ASPECT :
    Identifier le sentiment sur différents aspects d'un produit/service.

    Exemple : "La nourriture est excellente mais le service est lent"
    → nourriture = positif, service = négatif

    On utilise la classification zero-shot pour simuler cette tâche.
    """
    print("=" * 60)
    print("  Analyse de Sentiment par Aspect")
    print("=" * 60)

    classifieur = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    # Avis avec plusieurs aspects
    avis = "The hotel room was spacious and clean, but the breakfast was disappointing and the WiFi was extremely slow."

    # Extraire le sentiment pour chaque aspect
    aspects = {
        "room quality": "The hotel room was spacious and clean",
        "breakfast quality": "the breakfast was disappointing",
        "WiFi quality": "the WiFi was extremely slow",
    }

    sentiments = ["positive", "negative", "neutral"]

    print(f"\n  📝 Avis complet :")
    print(f"     \"{avis}\"\n")
    print(f"  Analyse par aspect :")
    print(f"  {'─' * 50}")

    for aspect, texte_aspect in aspects.items():
        result = classifieur(texte_aspect, sentiments)
        top_sentiment = result["labels"][0]
        top_score = result["scores"][0]

        if top_sentiment == "positive":
            emoji = "✅"
        elif top_sentiment == "negative":
            emoji = "❌"
        else:
            emoji = "➖"

        print(f"  {emoji} {aspect:20s} → {top_sentiment:8s} ({top_score:.4f})")
        print(f"     Extrait : \"{texte_aspect}\"")
        print()


# ============================================================
# SECTION 6 : Batch processing et statistiques
# ============================================================

def analyse_batch():
    """
    Analyse de sentiment en batch (traitement par lots).

    Utile quand on a beaucoup de textes à analyser
    (ex: analyser 1000 avis clients d'un coup).
    """
    print("=" * 60)
    print("  Analyse de Sentiment en Batch + Statistiques")
    print("=" * 60)

    analyseur = pipeline("sentiment-analysis", batch_size=8)

    # Simuler un lot d'avis clients
    avis_clients = [
        "Great product, fast delivery!",
        "Terrible quality, very disappointed.",
        "Works perfectly, exactly what I needed.",
        "Waste of money, broke in two days.",
        "Excellent customer service!",
        "Average product, nothing special.",
        "Love it! Best purchase ever.",
        "Horrible experience from start to finish.",
        "Good value for the price.",
        "The packaging was damaged but product is fine.",
        "Superb quality, highly recommend!",
        "Not worth the price at all.",
        "Decent, gets the job done.",
        "Outstanding performance and design!",
        "Mediocre at best, expected more.",
    ]

    # Analyser tous les avis d'un coup
    resultats = analyseur(avis_clients)

    # Compter les résultats
    positifs = []
    negatifs = []

    print(f"\n  Analyse de {len(avis_clients)} avis :\n")

    for avis, resultat in zip(avis_clients, resultats):
        label = resultat["label"]
        score = resultat["score"]

        if label == "POSITIVE":
            positifs.append(score)
            print(f"  😊 {score:.3f} | \"{avis}\"")
        else:
            negatifs.append(score)
            print(f"  😞 {score:.3f} | \"{avis}\"")

    # Statistiques globales
    print(f"\n  {'═' * 50}")
    print(f"  📊 STATISTIQUES GLOBALES")
    print(f"  {'═' * 50}")
    print(f"  Total avis analysés : {len(avis_clients)}")
    print()
    print(f"  😊 Positifs : {len(positifs)} ({len(positifs)/len(avis_clients)*100:.0f}%)")
    if positifs:
        print(f"     Confiance moyenne : {sum(positifs)/len(positifs):.4f}")
        print(f"     Confiance min     : {min(positifs):.4f}")
        print(f"     Confiance max     : {max(positifs):.4f}")
    print()
    print(f"  😞 Négatifs : {len(negatifs)} ({len(negatifs)/len(avis_clients)*100:.0f}%)")
    if negatifs:
        print(f"     Confiance moyenne : {sum(negatifs)/len(negatifs):.4f}")
        print(f"     Confiance min     : {min(negatifs):.4f}")
        print(f"     Confiance max     : {max(negatifs):.4f}")
    print()

    # Score de satisfaction global
    score_global = len(positifs) / len(avis_clients) * 5  # Score sur 5
    etoiles = "⭐" * int(round(score_global))
    print(f"  🏆 Score de satisfaction : {score_global:.1f}/5.0 {etoiles}")
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("█  PARTIE 1 - ÉTAPE 3 : ANALYSE DE SENTIMENT")
    print("█" * 60 + "\n")

    # 1. Analyse basique
    print("\n" + "─" * 60)
    print("  💬 SECTION 1 : Analyse Basique")
    print("─" * 60 + "\n")
    analyse_sentiment_basique()

    # 2. Multi-classe (étoiles)
    print("\n" + "─" * 60)
    print("  ⭐ SECTION 2 : Analyse Multi-Classe")
    print("─" * 60 + "\n")
    analyse_sentiment_multiclasse()

    # 3. Tweets
    print("\n" + "─" * 60)
    print("  🐦 SECTION 3 : Analyse de Tweets")
    print("─" * 60 + "\n")
    analyse_tweets()

    # 4. Analyse détaillée
    print("\n" + "─" * 60)
    print("  🔍 SECTION 4 : Analyse Détaillée")
    print("─" * 60 + "\n")
    analyse_sentiment_detaillee()

    # 5. Analyse par aspect
    print("\n" + "─" * 60)
    print("  🎯 SECTION 5 : Analyse par Aspect")
    print("─" * 60 + "\n")
    analyse_sentiment_aspect()

    # 6. Batch processing
    print("\n" + "─" * 60)
    print("  📦 SECTION 6 : Analyse en Batch")
    print("─" * 60 + "\n")
    analyse_batch()

    print("\n" + "█" * 60)
    print("█  FIN DE L'ANALYSE DE SENTIMENT")
    print("█" * 60 + "\n")
