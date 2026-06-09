# ============================================================
#  4_question_answering.py
#  Partie 1 - Étape 4 : Question Answering (QA)
# ============================================================
#  Ce script montre comment :
#    • Répondre à des questions à partir d'un contexte (Extractive QA)
#    • Utiliser différents modèles de QA
#    • Faire du QA manuellement (sans pipeline)
#    • QA sur des documents longs
# ============================================================

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline,
)


# ============================================================
# SECTION 1 : QA extractif avec Pipeline
# ============================================================

def qa_pipeline_basique():
    """
    Question Answering extractif avec pipeline.

    QA extractif = le modèle EXTRAIT la réponse directement
    du texte fourni (il ne génère pas de nouvelle réponse).

    Comment ça marche ?
      1. On donne un CONTEXTE (paragraphe de texte)
      2. On pose une QUESTION
      3. Le modèle trouve la RÉPONSE dans le contexte
    """
    print("=" * 60)
    print("  Question Answering - Pipeline Basique")
    print("=" * 60)

    # Créer la pipeline QA
    qa = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

    # Contexte : un paragraphe sur l'Intelligence Artificielle
    contexte = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create
    intelligent machines that can perform tasks that typically require human intelligence.
    The field was founded in 1956 at a conference at Dartmouth College. Modern AI includes
    machine learning, deep learning, natural language processing, and computer vision.
    Key figures in AI include Alan Turing, who proposed the Turing Test in 1950, and
    Geoffrey Hinton, who is known as the "Godfather of Deep Learning". Today, AI is used
    in various applications including virtual assistants like Siri and Alexa, self-driving
    cars developed by companies like Tesla and Waymo, and medical diagnosis systems.
    """

    # Questions à poser
    questions = [
        "What is Artificial Intelligence?",
        "When was the field of AI founded?",
        "Who is the Godfather of Deep Learning?",
        "What did Alan Turing propose in 1950?",
        "What companies develop self-driving cars?",
        "What are examples of virtual assistants?",
    ]

    print(f"\n  📄 Contexte :")
    print(f"  {contexte.strip()}\n")
    print(f"  {'─' * 50}")
    print(f"  ❓ Questions et Réponses :\n")

    for question in questions:
        resultat = qa(question=question, context=contexte)

        print(f"  ❓ {question}")
        print(f"  ✅ {resultat['answer']}")
        print(f"     Score : {resultat['score']:.4f}")
        print(f"     Position : [{resultat['start']}:{resultat['end']}]")
        print()


# ============================================================
# SECTION 2 : QA sur différents domaines
# ============================================================

def qa_multi_domaine():
    """
    QA sur des contextes de différents domaines :
    science, histoire, géographie, etc.
    """
    print("=" * 60)
    print("  Question Answering - Multi-Domaine")
    print("=" * 60)

    qa = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

    # Plusieurs contextes avec leurs questions
    domaines = [
        {
            "nom": "🔬 Science",
            "contexte": """
            DNA (Deoxyribonucleic acid) is a molecule that carries the genetic
            instructions for the development, functioning, growth, and reproduction
            of all known organisms. DNA was first identified by Friedrich Miescher
            in 1869. The structure of DNA, the famous double helix, was discovered
            by James Watson and Francis Crick in 1953 at Cambridge University,
            based on X-ray crystallography data from Rosalind Franklin.
            """,
            "questions": [
                "What does DNA stand for?",
                "Who discovered the structure of DNA?",
                "When was DNA first identified?",
                "Who provided X-ray crystallography data?",
            ]
        },
        {
            "nom": "🏛️ Histoire",
            "contexte": """
            The French Revolution began in 1789 and was a period of radical political
            and societal change in France. The revolution began with the convocation
            of the Estates General in May 1789. The storming of the Bastille on
            July 14, 1789 became a symbol of the revolution. King Louis XVI was
            executed by guillotine on January 21, 1793. The revolution ended with
            Napoleon Bonaparte's coup d'état in November 1799, establishing the
            French Consulate.
            """,
            "questions": [
                "When did the French Revolution begin?",
                "What happened on July 14, 1789?",
                "When was King Louis XVI executed?",
                "Who ended the revolution?",
            ]
        },
        {
            "nom": "🌍 Géographie",
            "contexte": """
            Morocco is a country located in North Africa, bordered by the Atlantic
            Ocean and the Mediterranean Sea. The capital city is Rabat, while the
            largest city is Casablanca. Morocco has a population of approximately
            37 million people. The official languages are Arabic and Amazigh.
            Mount Toubkal, at 4,167 meters, is the highest peak in North Africa.
            The country is known for its diverse landscape, from the Sahara Desert
            to the Atlas Mountains and beautiful coastal cities.
            """,
            "questions": [
                "What is the capital of Morocco?",
                "What is the largest city?",
                "How high is Mount Toubkal?",
                "What are the official languages?",
            ]
        },
    ]

    for domaine in domaines:
        print(f"\n  {domaine['nom']}")
        print(f"  {'─' * 50}")

        for question in domaine["questions"]:
            resultat = qa(
                question=question,
                context=domaine["contexte"]
            )
            print(f"    ❓ {question}")
            print(f"    ✅ {resultat['answer']} (score: {resultat['score']:.4f})")
        print()


# ============================================================
# SECTION 3 : QA manuel (sans pipeline)
# ============================================================

def qa_manuel():
    """
    Question Answering MANUEL pour comprendre le fonctionnement interne.

    Le modèle de QA fait ceci :
      1. Tokenize la question + le contexte ensemble
      2. Produit 2 vecteurs de scores :
         - start_logits : probabilité que chaque token soit le DÉBUT
         - end_logits   : probabilité que chaque token soit la FIN
      3. La réponse = les tokens entre start et end
    """
    print("=" * 60)
    print("  Question Answering Manuel (sous le capot)")
    print("=" * 60)

    nom_modele = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(nom_modele)
    model = AutoModelForQuestionAnswering.from_pretrained(nom_modele)

    contexte = "The Eiffel Tower is located in Paris, France. It was built in 1889 for the World Fair."
    question = "When was the Eiffel Tower built?"

    print(f"\n  Contexte  : \"{contexte}\"")
    print(f"  Question  : \"{question}\"\n")

    # Étape 1 : Tokenization
    inputs = tokenizer(
        question,
        contexte,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"  Étape 1 - Tokenization :")
    print(f"    Nombre de tokens : {len(tokens)}")
    print(f"    Tokens : {tokens}")
    print()

    # Étape 2 : Prédiction du modèle
    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    print(f"  Étape 2 - Scores du modèle :")
    print(f"    start_logits shape : {start_logits.shape}")
    print(f"    end_logits shape   : {end_logits.shape}")
    print()

    # Étape 3 : Trouver les positions start et end
    start_probs = torch.nn.functional.softmax(start_logits, dim=-1)
    end_probs = torch.nn.functional.softmax(end_logits, dim=-1)

    start_idx = torch.argmax(start_probs).item()
    end_idx = torch.argmax(end_probs).item()

    print(f"  Étape 3 - Positions de la réponse :")
    print(f"    Position début : {start_idx} (token: '{tokens[start_idx]}')")
    print(f"    Position fin   : {end_idx} (token: '{tokens[end_idx]}')")
    print()

    # Afficher les scores pour chaque token
    print(f"  Top 5 tokens pour le DÉBUT :")
    top_start = torch.topk(start_probs, 5)
    for score, idx in zip(top_start.values, top_start.indices):
        print(f"    '{tokens[idx]:15s}' → {score:.4f}")
    print()

    print(f"  Top 5 tokens pour la FIN :")
    top_end = torch.topk(end_probs, 5)
    for score, idx in zip(top_end.values, top_end.indices):
        print(f"    '{tokens[idx]:15s}' → {score:.4f}")
    print()

    # Étape 4 : Extraire la réponse
    reponse_tokens = tokens[start_idx:end_idx + 1]
    reponse = tokenizer.decode(
        inputs["input_ids"][0][start_idx:end_idx + 1]
    )

    print(f"  Étape 4 - Réponse extraite :")
    print(f"    Tokens de la réponse : {reponse_tokens}")
    print(f"    Réponse finale : \"{reponse}\"")
    print()


# ============================================================
# SECTION 4 : QA avec score de confiance
# ============================================================

def qa_avec_confiance():
    """
    QA avec analyse de la confiance du modèle.

    Quand le modèle n'est pas sûr de sa réponse, le score est bas.
    On peut utiliser ce score pour filtrer les mauvaises réponses.
    """
    print("=" * 60)
    print("  Question Answering avec Analyse de Confiance")
    print("=" * 60)

    qa = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

    contexte = """
    Python is a high-level programming language created by Guido van Rossum
    and first released in 1991. Python emphasizes code readability with its
    use of significant indentation. It supports multiple programming paradigms
    including procedural, object-oriented, and functional programming.
    Python is widely used in web development, data science, artificial
    intelligence, and automation.
    """

    # Questions avec différents niveaux de difficulté
    questions = [
        # Facile - réponse directement dans le texte
        ("Who created Python?", "facile"),
        ("When was Python first released?", "facile"),
        # Moyen - nécessite un peu d'inférence
        ("What paradigms does Python support?", "moyen"),
        # Difficile - réponse pas vraiment dans le texte
        ("What is the latest version of Python?", "difficile"),
        ("Is Python faster than Java?", "difficile"),
    ]

    print(f"\n  Seuil de confiance : 0.30\n")

    for question, difficulte in questions:
        resultat = qa(question=question, context=contexte)
        score = resultat["score"]
        reponse = resultat["answer"]

        # Indicateur de confiance
        if score > 0.7:
            indicateur = "🟢 Haute"
        elif score > 0.3:
            indicateur = "🟡 Moyenne"
        else:
            indicateur = "🔴 Basse"

        fiable = "✅ Fiable" if score > 0.30 else "⚠️ Non fiable"

        print(f"  [{difficulte:10s}] ❓ {question}")
        print(f"              ✅ \"{reponse}\"")
        print(f"              📊 Score: {score:.4f} | {indicateur} | {fiable}")
        print()


# ============================================================
# SECTION 5 : QA sur des textes longs (avec chunking)
# ============================================================

def qa_texte_long():
    """
    QA sur des textes plus longs que la limite du modèle.

    Les modèles Transformers ont une limite de tokens (512 ou 1024).
    Pour les textes longs, on les découpe en morceaux (chunks)
    et on cherche la réponse dans chaque morceau.

    La pipeline gère cela automatiquement avec le paramètre
    doc_stride (chevauchement entre les morceaux).
    """
    print("=" * 60)
    print("  Question Answering sur Textes Longs")
    print("=" * 60)

    qa = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

    # Texte long (simulé)
    texte_long = """
    Natural Language Processing (NLP) is a subfield of artificial intelligence
    and computational linguistics that focuses on the interaction between
    computers and human language. The history of NLP goes back to the 1950s
    when Alan Turing published his famous article "Computing Machinery and
    Intelligence". In the early days, NLP systems were based on hand-crafted
    rules and patterns.

    The introduction of statistical methods in the 1990s marked a significant
    shift in NLP. Researchers began using large corpora of text to train
    probabilistic models. The most notable advancement was the development
    of n-gram language models and Hidden Markov Models (HMMs) for tasks
    like part-of-speech tagging and named entity recognition.

    The deep learning revolution, starting around 2013 with Word2Vec by
    Tomas Mikolov at Google, transformed NLP. Word2Vec introduced the concept
    of word embeddings, where words are represented as dense vectors in a
    continuous space. This was followed by GloVe (Global Vectors) from
    Stanford University in 2014.

    The most significant breakthrough came in 2017 with the paper "Attention
    is All You Need" by Vaswani et al., which introduced the Transformer
    architecture. This architecture uses self-attention mechanisms instead
    of recurrence, enabling parallel processing and better handling of
    long-range dependencies.

    BERT (Bidirectional Encoder Representations from Transformers) was
    released by Google in 2018 and achieved state-of-the-art results on
    many NLP benchmarks. BERT is pre-trained on a large corpus using two
    tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).

    GPT (Generative Pre-trained Transformer) by OpenAI took a different
    approach, using a decoder-only architecture for text generation. GPT-3,
    released in 2020, has 175 billion parameters and demonstrated impressive
    few-shot learning capabilities.
    """

    questions = [
        "Who created Word2Vec?",
        "What did the 2017 paper introduce?",
        "How many parameters does GPT-3 have?",
        "What tasks is BERT pre-trained on?",
        "When was BERT released?",
    ]

    print(f"\n  📄 Texte : {len(texte_long.split())} mots\n")

    for question in questions:
        resultat = qa(
            question=question,
            context=texte_long,
        )
        print(f"  ❓ {question}")
        print(f"  ✅ {resultat['answer']} (score: {resultat['score']:.4f})")
        print()


# ============================================================
# SECTION 6 : QA en français
# ============================================================

def qa_francais():
    """
    Question Answering en français avec un modèle multilingue.
    """
    print("=" * 60)
    print("  Question Answering en Français")
    print("=" * 60)

    # Modèle multilingue pour QA
    qa = pipeline(
        "question-answering",
        model="deepset/xlm-roberta-large-squad2"
    )

    contexte_fr = """
    L'École Nationale des Sciences Appliquées (ENSA) d'Al Hoceima est un
    établissement d'enseignement supérieur public marocain. Elle a été créée
    en 2007 et fait partie du réseau des ENSA du Maroc. L'école forme des
    ingénieurs d'état dans plusieurs filières dont le Génie Informatique,
    le Génie Civil et le Génie Électrique. La durée de formation est de
    5 ans après le baccalauréat, comprenant 2 années de cycle préparatoire
    intégré et 3 années de cycle ingénieur.
    """

    questions_fr = [
        "Quand a été créée l'ENSA d'Al Hoceima ?",
        "Quelle est la durée de formation ?",
        "Quelles filières sont proposées ?",
        "Combien d'années dure le cycle préparatoire ?",
    ]

    print(f"\n  📄 Contexte (FR) :")
    print(f"  {contexte_fr.strip()}\n")
    print(f"  {'─' * 50}\n")

    for question in questions_fr:
        resultat = qa(question=question, context=contexte_fr)
        print(f"  ❓ {question}")
        print(f"  ✅ {resultat['answer']} (score: {resultat['score']:.4f})")
        print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("█  PARTIE 1 - ÉTAPE 4 : QUESTION ANSWERING")
    print("█" * 60 + "\n")

    # 1. QA basique
    print("\n" + "─" * 60)
    print("  💬 SECTION 1 : QA avec Pipeline")
    print("─" * 60 + "\n")
    qa_pipeline_basique()

    # 2. Multi-domaine
    print("\n" + "─" * 60)
    print("  🌐 SECTION 2 : QA Multi-Domaine")
    print("─" * 60 + "\n")
    qa_multi_domaine()

    # 3. QA manuel
    print("\n" + "─" * 60)
    print("  🔧 SECTION 3 : QA Manuel (sous le capot)")
    print("─" * 60 + "\n")
    qa_manuel()

    # 4. QA avec confiance
    print("\n" + "─" * 60)
    print("  📊 SECTION 4 : QA avec Confiance")
    print("─" * 60 + "\n")
    qa_avec_confiance()

    # 5. Textes longs
    print("\n" + "─" * 60)
    print("  📚 SECTION 5 : QA sur Textes Longs")
    print("─" * 60 + "\n")
    qa_texte_long()

    # 6. QA en français
    print("\n" + "─" * 60)
    print("  🇫🇷 SECTION 6 : QA en Français")
    print("─" * 60 + "\n")
    qa_francais()

    print("\n" + "█" * 60)
    print("█  FIN DU QUESTION ANSWERING")
    print("█" * 60 + "\n")
