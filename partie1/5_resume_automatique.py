# ============================================================
#  5_resume_automatique.py
#  Partie 1 - Étape 5 : Résumé Automatique
# ============================================================
#  Ce script montre comment :
#    • Générer des résumés automatiques de textes
#    • Résumé extractif vs abstractif
#    • Contrôler la longueur du résumé
#    • Résumer des articles, documents, etc.
# ============================================================

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)


# ============================================================
# SECTION 1 : Résumé avec Pipeline (méthode simple)
# ============================================================

def resume_pipeline_basique():
    """
    Résumé automatique avec la pipeline HuggingFace.

    Il existe 2 types de résumé :
      - Extractif : sélectionne les phrases les plus importantes
      - Abstractif : génère de NOUVELLES phrases qui résument le texte

    HuggingFace utilise des modèles ABSTRACTIFS (ex: BART, T5, Pegasus)
    qui comprennent le texte et génèrent un nouveau résumé.
    """
    print("=" * 60)
    print("  Résumé Automatique - Pipeline Basique")
    print("=" * 60)

    # Créer la pipeline de résumé
    resumeur = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    # Article d'exemple
    article = """
    Artificial intelligence has made remarkable progress in recent years,
    transforming industries from healthcare to finance. Machine learning
    algorithms can now diagnose diseases from medical images with accuracy
    rivaling human doctors. In the financial sector, AI-powered trading
    systems analyze market data in milliseconds, making decisions that
    would take human traders hours. Natural language processing has enabled
    the development of sophisticated chatbots and virtual assistants that
    can understand and respond to human queries in natural language.
    However, these advancements also raise important ethical questions.
    Concerns about bias in AI systems, job displacement, and privacy
    have sparked global debates. Governments around the world are now
    working on regulations to ensure AI is developed and deployed
    responsibly. Experts emphasize the need for transparent AI systems
    that can explain their decisions. Despite these challenges, the
    potential benefits of AI are enormous, and continued research and
    development are expected to bring even more breakthroughs in the
    coming years.
    """

    print(f"\n  📄 Article original ({len(article.split())} mots) :")
    print(f"  {'─' * 50}")
    print(f"  {article.strip()}")
    print(f"  {'─' * 50}\n")

    # Générer le résumé
    resume = resumeur(
        article,
        max_length=80,     # Longueur max du résumé
        min_length=30,     # Longueur min du résumé
        do_sample=False,   # Pas d'échantillonnage (déterministe)
    )[0]["summary_text"]

    print(f"  📝 Résumé généré ({len(resume.split())} mots) :")
    print(f"  {'─' * 50}")
    print(f"  {resume}")
    print(f"  {'─' * 50}")
    print(f"\n  📊 Taux de compression : {len(resume.split())/len(article.split())*100:.0f}%")
    print()


# ============================================================
# SECTION 2 : Contrôle de la longueur du résumé
# ============================================================

def resume_longueur_variable():
    """
    Montre comment contrôler la longueur du résumé :
      - Résumé court (1-2 phrases)
      - Résumé moyen (3-4 phrases)
      - Résumé long (paragraphe)
    """
    print("=" * 60)
    print("  Résumé à Longueur Variable")
    print("=" * 60)

    resumeur = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    article = """
    Climate change represents one of the most significant challenges facing
    humanity today. Global temperatures have risen by approximately 1.1 degrees
    Celsius since pre-industrial times, primarily due to the burning of fossil
    fuels and deforestation. The effects are already visible: rising sea levels
    threaten coastal communities, extreme weather events are becoming more
    frequent and severe, and ecosystems are being disrupted worldwide.

    Scientists have established that limiting warming to 1.5 degrees Celsius
    above pre-industrial levels is critical to avoiding the worst impacts.
    This requires reducing global carbon dioxide emissions by 45 percent
    from 2010 levels by 2030 and reaching net zero by 2050. To achieve
    these targets, governments must implement ambitious policies including
    transitioning to renewable energy sources, improving energy efficiency,
    and protecting natural carbon sinks like forests and oceans.

    The economic implications are substantial. While the transition to a
    low-carbon economy requires significant investment, it also presents
    enormous opportunities. The renewable energy sector is already creating
    millions of jobs worldwide. Countries that lead in clean technology
    innovation stand to gain competitive advantages in the global economy.
    International cooperation, such as the Paris Agreement signed in 2015,
    remains essential for coordinating global efforts to combat climate change.
    """

    configurations = [
        {"nom": "Court   (1-2 phrases)", "max": 40, "min": 15},
        {"nom": "Moyen   (3-4 phrases)", "max": 80, "min": 40},
        {"nom": "Long    (paragraphe) ", "max": 150, "min": 80},
    ]

    print(f"\n  📄 Article original : {len(article.split())} mots\n")

    for config in configurations:
        resume = resumeur(
            article,
            max_length=config["max"],
            min_length=config["min"],
            do_sample=False,
        )[0]["summary_text"]

        taux = len(resume.split()) / len(article.split()) * 100

        print(f"  📝 {config['nom']} ({len(resume.split())} mots, {taux:.0f}%) :")
        print(f"     {resume}")
        print()


# ============================================================
# SECTION 3 : Résumé avec différents modèles
# ============================================================

def resume_differents_modeles():
    """
    Compare les résumés produits par différents modèles :
      - BART (Facebook) : bon pour les articles de news
      - T5 (Google)     : modèle polyvalent
      - DistilBART      : version légère de BART (plus rapide)
    """
    print("=" * 60)
    print("  Comparaison de Modèles de Résumé")
    print("=" * 60)

    article = """
    SpaceX, founded by Elon Musk in 2002, has revolutionized the space industry
    with its reusable rocket technology. The Falcon 9 rocket can land itself
    after launching payloads into orbit, drastically reducing the cost of space
    launches. The company's Starship vehicle, currently in development, aims to
    carry humans to Mars. SpaceX has also launched thousands of Starlink
    satellites to provide global internet coverage. The company has secured
    numerous contracts with NASA, including transporting astronauts to the
    International Space Station through its Crew Dragon capsule.
    """

    modeles = [
        ("sshleifer/distilbart-cnn-12-6", "DistilBART (rapide)"),
        ("facebook/bart-large-cnn", "BART-Large (qualité)"),
    ]

    print(f"\n  📄 Article original ({len(article.split())} mots) :\n")
    print(f"  {article.strip()}\n")

    for nom_modele, description in modeles:
        print(f"  {'─' * 50}")
        print(f"  🤖 Modèle : {description}")
        print(f"     ({nom_modele})")

        resumeur = pipeline("summarization", model=nom_modele)
        resume = resumeur(
            article,
            max_length=60,
            min_length=20,
            do_sample=False,
        )[0]["summary_text"]

        print(f"     Résumé ({len(resume.split())} mots) :")
        print(f"     {resume}")
        print()


# ============================================================
# SECTION 4 : Résumé avec T5 (Text-to-Text)
# ============================================================

def resume_t5():
    """
    Résumé avec le modèle T5 (Text-to-Text Transfer Transformer).

    T5 traite TOUTES les tâches NLP comme des problèmes de
    texte-vers-texte. Pour le résumé, on ajoute le préfixe
    "summarize: " devant le texte.

    Exemples de préfixes T5 :
      - "summarize: ..."      → résumé
      - "translate English to French: ..." → traduction
      - "question: ... context: ..."       → QA
    """
    print("=" * 60)
    print("  Résumé avec T5 (Text-to-Text)")
    print("=" * 60)

    # Charger T5 manuellement pour montrer le préfixe
    nom_modele = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(nom_modele)
    model = AutoModelForSeq2SeqLM.from_pretrained(nom_modele)

    article = """
    The World Health Organization declared COVID-19 a global pandemic in March
    2020. The virus, caused by SARS-CoV-2, spread rapidly across the globe,
    leading to lockdowns and social distancing measures in most countries.
    Scientists developed multiple vaccines in record time, with the first
    vaccines receiving emergency authorization in December 2020. The pandemic
    highlighted the importance of public health infrastructure, international
    cooperation, and scientific research in addressing global health crises.
    """

    # Le préfixe "summarize:" dit à T5 de faire un résumé
    texte_input = "summarize: " + article.strip()

    print(f"\n  📄 Texte d'entrée (avec préfixe T5) :")
    print(f"     \"summarize: [article de {len(article.split())} mots]\"\n")

    # Tokenizer
    inputs = tokenizer(
        texte_input,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    # Générer le résumé
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=80,
            min_length=20,
            num_beams=4,          # Beam search pour meilleure qualité
            length_penalty=2.0,   # Pénaliser les résumés trop courts
            early_stopping=True,
        )

    resume = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"  📝 Résumé T5 ({len(resume.split())} mots) :")
    print(f"     {resume}")
    print()

    # Montrer les paramètres de génération
    print(f"  ⚙️ Paramètres de génération :")
    print(f"     num_beams      = 4 (recherche en faisceau)")
    print(f"     length_penalty = 2.0 (favoriser les résumés plus longs)")
    print(f"     early_stopping = True (arrêter quand tous les faisceaux finissent)")
    print()


# ============================================================
# SECTION 5 : Résumé de plusieurs documents
# ============================================================

def resume_multi_documents():
    """
    Résumer plusieurs documents / paragraphes individuellement
    puis combiner les résumés.

    Utile pour :
      - Résumer un rapport de plusieurs pages
      - Résumer une collection d'articles
      - Créer un digest de news
    """
    print("=" * 60)
    print("  Résumé Multi-Documents")
    print("=" * 60)

    resumeur = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    documents = [
        {
            "titre": "Article 1 : L'IA dans la Santé",
            "texte": """
            Artificial intelligence is transforming healthcare in unprecedented ways.
            Machine learning algorithms can now analyze medical images, including
            X-rays, MRIs, and CT scans, with accuracy comparable to experienced
            radiologists. AI-powered diagnostic tools can detect diseases like cancer,
            diabetic retinopathy, and cardiovascular conditions at early stages when
            treatment is most effective. Additionally, AI is being used to accelerate
            drug discovery, with models that can predict molecular properties and
            identify potential drug candidates in a fraction of the time required
            by traditional methods.
            """
        },
        {
            "titre": "Article 2 : L'IA dans l'Éducation",
            "texte": """
            The education sector is experiencing a revolution driven by artificial
            intelligence. Adaptive learning platforms use AI to personalize the
            learning experience for each student, adjusting difficulty levels and
            content based on individual performance. AI tutors can provide instant
            feedback on assignments and identify areas where students struggle.
            Language learning apps powered by NLP technology enable conversation
            practice with virtual partners. Furthermore, AI is helping educators
            automate administrative tasks such as grading and attendance tracking,
            allowing them to focus more on teaching.
            """
        },
        {
            "titre": "Article 3 : L'IA dans les Transports",
            "texte": """
            Autonomous vehicles represent one of the most visible applications of AI
            in transportation. Companies like Tesla, Waymo, and Cruise are developing
            self-driving cars that use computer vision, lidar, and deep learning to
            navigate roads safely. Beyond personal vehicles, AI is optimizing public
            transportation systems by predicting demand, optimizing routes, and
            reducing delays. Logistics companies use AI-powered algorithms to plan
            delivery routes, reducing fuel consumption and delivery times. Air traffic
            management systems also benefit from AI for safer and more efficient
            flight operations.
            """
        },
    ]

    print()
    resumes_individuels = []

    for doc in documents:
        resume = resumeur(
            doc["texte"],
            max_length=50,
            min_length=20,
            do_sample=False,
        )[0]["summary_text"]

        resumes_individuels.append(resume)

        print(f"  📄 {doc['titre']}")
        print(f"     Original  : {len(doc['texte'].split())} mots")
        print(f"     Résumé    : {resume}")
        print()

    # Résumé global (résumer les résumés)
    texte_combine = " ".join(resumes_individuels)

    resume_global = resumeur(
        texte_combine,
        max_length=60,
        min_length=25,
        do_sample=False,
    )[0]["summary_text"]

    print(f"  {'═' * 50}")
    print(f"  📋 RÉSUMÉ GLOBAL ({len(resume_global.split())} mots) :")
    print(f"     {resume_global}")
    print()


# ============================================================
# SECTION 6 : Résumé extractif (simple)
# ============================================================

def resume_extractif_simple():
    """
    Résumé EXTRACTIF simple (sans modèle transformer).

    Le résumé extractif sélectionne les phrases les plus
    importantes du texte original (ne génère pas de nouveau texte).

    Méthode simple : scoring basé sur la fréquence des mots.
      1. Compter la fréquence de chaque mot
      2. Scorer chaque phrase selon ses mots importants
      3. Sélectionner les phrases avec le meilleur score
    """
    print("=" * 60)
    print("  Résumé Extractif Simple (basé sur fréquence)")
    print("=" * 60)

    texte = """
    Deep learning has emerged as a powerful subset of machine learning that uses
    neural networks with multiple layers to model complex patterns in data.
    Convolutional Neural Networks (CNNs) have revolutionized computer vision,
    enabling accurate image classification, object detection, and image segmentation.
    Recurrent Neural Networks (RNNs) and their variants like LSTM have been widely
    used for sequential data such as text and time series. The introduction of
    Transformers in 2017 marked a paradigm shift, replacing recurrence with
    self-attention mechanisms. Models like BERT and GPT have achieved remarkable
    results across many natural language processing tasks. Transfer learning,
    where models pre-trained on large datasets are fine-tuned for specific tasks,
    has become the dominant approach in both NLP and computer vision.
    """

    # Découper en phrases
    import re
    phrases = [p.strip() for p in re.split(r'(?<=[.!?])\s+', texte.strip()) if p.strip()]

    # Compter la fréquence des mots
    mots = texte.lower().split()
    # Exclure les mots communs (stop words simplifiés)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "has", "have",
                  "had", "in", "on", "at", "to", "for", "of", "and", "or",
                  "with", "that", "this", "their", "its", "as", "by", "from",
                  "been", "such", "like", "where"}
    mots_filtres = [m.strip(".,!?;:") for m in mots if m.strip(".,!?;:") not in stop_words]

    # Fréquence
    freq = {}
    for mot in mots_filtres:
        freq[mot] = freq.get(mot, 0) + 1

    # Normaliser
    max_freq = max(freq.values()) if freq else 1
    for mot in freq:
        freq[mot] /= max_freq

    # Scorer chaque phrase
    scores = []
    for phrase in phrases:
        mots_phrase = phrase.lower().split()
        score = sum(freq.get(m.strip(".,!?;:"), 0) for m in mots_phrase)
        score /= max(len(mots_phrase), 1)  # Normaliser par la longueur
        scores.append((score, phrase))

    # Trier par score
    scores.sort(reverse=True)

    print(f"\n  📄 Texte original : {len(phrases)} phrases\n")

    print(f"  📊 Scores des phrases :")
    for i, (score, phrase) in enumerate(scores, 1):
        barre = "█" * int(score * 30)
        print(f"    {i}. [{score:.3f}] |{barre}")
        print(f"       \"{phrase[:80]}...\"" if len(phrase) > 80 else f"       \"{phrase}\"")
        print()

    # Sélectionner les 3 meilleures phrases
    nb_phrases_resume = 3
    meilleures = scores[:nb_phrases_resume]

    print(f"  {'═' * 50}")
    print(f"  📝 Résumé extractif (top {nb_phrases_resume} phrases) :")
    for _, phrase in meilleures:
        print(f"     • {phrase}")
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("█  PARTIE 1 - ÉTAPE 5 : RÉSUMÉ AUTOMATIQUE")
    print("█" * 60 + "\n")

    # 1. Résumé basique
    print("\n" + "─" * 60)
    print("  📝 SECTION 1 : Résumé avec Pipeline")
    print("─" * 60 + "\n")
    resume_pipeline_basique()

    # 2. Longueur variable
    print("\n" + "─" * 60)
    print("  📏 SECTION 2 : Longueur Variable")
    print("─" * 60 + "\n")
    resume_longueur_variable()

    # 3. Différents modèles
    print("\n" + "─" * 60)
    print("  🤖 SECTION 3 : Comparaison de Modèles")
    print("─" * 60 + "\n")
    resume_differents_modeles()

    # 4. T5
    print("\n" + "─" * 60)
    print("  🔤 SECTION 4 : Résumé avec T5")
    print("─" * 60 + "\n")
    resume_t5()

    # 5. Multi-documents
    print("\n" + "─" * 60)
    print("  📚 SECTION 5 : Résumé Multi-Documents")
    print("─" * 60 + "\n")
    resume_multi_documents()

    # 6. Extractif
    print("\n" + "─" * 60)
    print("  ✂️ SECTION 6 : Résumé Extractif Simple")
    print("─" * 60 + "\n")
    resume_extractif_simple()

    print("\n" + "█" * 60)
    print("█  FIN DU RÉSUMÉ AUTOMATIQUE")
    print("█" * 60 + "\n")
