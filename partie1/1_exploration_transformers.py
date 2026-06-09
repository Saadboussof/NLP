# ============================================================
#  1_exploration_transformers.py
#  Partie 1 - Étape 1 : Explorer la bibliothèque HuggingFace
# ============================================================
#  Ce script montre comment :
#    • Charger un modèle pré-entraîné (BERT, GPT-2, etc.)
#    • Utiliser un tokenizer pour convertir du texte en tokens
#    • Utiliser les pipelines simplifiées de HuggingFace
# ============================================================

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertModel,
    BertTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    pipeline,
)


# ============================================================
# SECTION 1 : Chargement de modèles pré-entraînés
# ============================================================

def charger_modele_bert():
    """
    Charge le modèle BERT (bert-base-uncased) et son tokenizer.

    BERT = Bidirectional Encoder Representations from Transformers
    - C'est un modèle "encoder-only" (il lit le texte dans les 2 sens)
    - Idéal pour : classification, NER, question answering
    - Taille : ~110 millions de paramètres

    Returns:
        tuple: (modèle BERT, tokenizer BERT)
    """
    print("=" * 60)
    print("  Chargement du modèle BERT (bert-base-uncased)")
    print("=" * 60)

    # Le tokenizer transforme le texte en nombres (tokens)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Le modèle contient les poids pré-entraînés
    model = BertModel.from_pretrained("bert-base-uncased")

    # Afficher les informations du modèle
    nb_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Modèle chargé avec succès")
    print(f"  ✓ Nombre de paramètres : {nb_params:,}")
    print(f"  ✓ Taille du vocabulaire : {tokenizer.vocab_size:,}")
    print(f"  ✓ Longueur max des séquences : {tokenizer.model_max_length}")
    print()

    return model, tokenizer


def charger_modele_gpt2():
    """
    Charge le modèle GPT-2 et son tokenizer.

    GPT-2 = Generative Pre-trained Transformer 2
    - C'est un modèle "decoder-only" (il génère du texte mot par mot)
    - Idéal pour : génération de texte, complétion
    - Taille : ~124 millions de paramètres (version small)

    Returns:
        tuple: (modèle GPT-2, tokenizer GPT-2)
    """
    print("=" * 60)
    print("  Chargement du modèle GPT-2")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")

    nb_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Modèle chargé avec succès")
    print(f"  ✓ Nombre de paramètres : {nb_params:,}")
    print(f"  ✓ Taille du vocabulaire : {tokenizer.vocab_size:,}")
    print()

    return model, tokenizer


def charger_modele_auto(nom_modele: str):
    """
    Charge N'IMPORTE QUEL modèle avec AutoModel / AutoTokenizer.

    La classe "Auto" détecte automatiquement le type de modèle
    (BERT, GPT-2, RoBERTa, T5, etc.) à partir du nom.

    Args:
        nom_modele: Nom du modèle sur HuggingFace Hub
                    Exemples : "bert-base-uncased", "gpt2",
                               "distilbert-base-uncased"

    Returns:
        tuple: (modèle, tokenizer)
    """
    print("=" * 60)
    print(f"  Chargement automatique : {nom_modele}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(nom_modele)
    model = AutoModel.from_pretrained(nom_modele)

    nb_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Type du modèle : {type(model).__name__}")
    print(f"  ✓ Nombre de paramètres : {nb_params:,}")
    print()

    return model, tokenizer


# ============================================================
# SECTION 2 : Utilisation des Tokenizers
# ============================================================

def explorer_tokenizer():
    """
    Montre en détail comment fonctionne un tokenizer.

    Un tokenizer fait 3 choses :
      1. Découpe le texte en morceaux (tokens)
      2. Convertit chaque token en un nombre (ID)
      3. Ajoute des tokens spéciaux ([CLS], [SEP], etc.)
    """
    print("=" * 60)
    print("  Exploration du Tokenizer BERT")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Texte d'exemple
    texte = "Bonjour, je suis un étudiant en Intelligence Artificielle à l'ENSA Al Hoceima."

    print(f"\n  Texte original :")
    print(f"  → \"{texte}\"\n")

    # ---- Étape 1 : Tokenization simple ----
    tokens = tokenizer.tokenize(texte)
    print(f"  Étape 1 - Tokenization (découpage en tokens) :")
    print(f"  → {tokens}")
    print(f"  → Nombre de tokens : {len(tokens)}\n")

    # ---- Étape 2 : Conversion en IDs ----
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"  Étape 2 - Conversion tokens → IDs :")
    for token, tid in zip(tokens, token_ids):
        print(f"     '{token}' → {tid}")
    print()

    # ---- Étape 3 : Encodage complet (avec tokens spéciaux) ----
    encoded = tokenizer.encode_plus(
        texte,
        add_special_tokens=True,     # Ajouter [CLS] et [SEP]
        max_length=128,              # Longueur maximale
        padding="max_length",        # Remplir jusqu'à max_length
        truncation=True,             # Couper si trop long
        return_attention_mask=True,  # Masque d'attention
        return_tensors="pt",         # Retourner des tenseurs PyTorch
    )

    print(f"  Étape 3 - Encodage complet :")
    print(f"  → input_ids shape      : {encoded['input_ids'].shape}")
    print(f"  → attention_mask shape  : {encoded['attention_mask'].shape}")
    print(f"  → Premiers 20 IDs      : {encoded['input_ids'][0][:20].tolist()}")
    print()

    # ---- Étape 4 : Décodage (retour au texte) ----
    texte_decode = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
    print(f"  Étape 4 - Décodage (IDs → texte) :")
    print(f"  → \"{texte_decode}\"")
    print()

    # ---- Étape 5 : Tokens spéciaux ----
    print(f"  Tokens spéciaux de BERT :")
    print(f"  → [CLS] = {tokenizer.cls_token} (ID: {tokenizer.cls_token_id}) - Début de séquence")
    print(f"  → [SEP] = {tokenizer.sep_token} (ID: {tokenizer.sep_token_id}) - Séparateur")
    print(f"  → [PAD] = {tokenizer.pad_token} (ID: {tokenizer.pad_token_id}) - Remplissage")
    print(f"  → [UNK] = {tokenizer.unk_token} (ID: {tokenizer.unk_token_id}) - Inconnu")
    print(f"  → [MASK]= {tokenizer.mask_token} (ID: {tokenizer.mask_token_id}) - Masque")
    print()

    return tokenizer


def comparer_tokenizers():
    """
    Compare comment BERT et GPT-2 tokenisent le même texte.

    BERT utilise WordPiece (découpe les mots rares en sous-mots)
    GPT-2 utilise BPE (Byte Pair Encoding)
    """
    print("=" * 60)
    print("  Comparaison des Tokenizers : BERT vs GPT-2")
    print("=" * 60)

    texte = "Natural Language Processing is fascinating!"

    # Tokenizer BERT (WordPiece)
    tok_bert = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens_bert = tok_bert.tokenize(texte)

    # Tokenizer GPT-2 (BPE)
    tok_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
    tokens_gpt2 = tok_gpt2.tokenize(texte)

    print(f"\n  Texte : \"{texte}\"\n")
    print(f"  BERT  (WordPiece) : {tokens_bert}")
    print(f"    → {len(tokens_bert)} tokens\n")
    print(f"  GPT-2 (BPE)       : {tokens_gpt2}")
    print(f"    → {len(tokens_gpt2)} tokens\n")

    # Texte avec un mot rare pour voir la différence
    texte_rare = "Electroencephalography is a neuroimaging technique."
    tokens_bert_rare = tok_bert.tokenize(texte_rare)
    tokens_gpt2_rare = tok_gpt2.tokenize(texte_rare)

    print(f"  Texte rare : \"{texte_rare}\"\n")
    print(f"  BERT  : {tokens_bert_rare}")
    print(f"  GPT-2 : {tokens_gpt2_rare}")
    print()


# ============================================================
# SECTION 3 : Pipelines simplifiées
# ============================================================

def demo_pipelines():
    """
    Montre les pipelines HuggingFace : la façon la PLUS SIMPLE
    d'utiliser les modèles Transformers.

    Une pipeline = tokenizer + modèle + post-traitement
    en UNE SEULE ligne de code !
    """
    print("=" * 60)
    print("  Démonstration des Pipelines HuggingFace")
    print("=" * 60)

    # ---- Pipeline 1 : Analyse de sentiment ----
    print("\n  ▶ Pipeline : Analyse de Sentiment")
    print("  " + "-" * 40)
    classifieur = pipeline("sentiment-analysis")

    textes = [
        "I love this movie, it's absolutely amazing!",
        "This is the worst experience I have ever had.",
        "The weather is okay today, nothing special.",
    ]

    for texte in textes:
        resultat = classifieur(texte)[0]
        emoji = "😊" if resultat["label"] == "POSITIVE" else "😞"
        print(f"  {emoji} \"{texte}\"")
        print(f"     → {resultat['label']} (confiance: {resultat['score']:.4f})")
    print()

    # ---- Pipeline 2 : Remplissage de masque (Fill-Mask) ----
    print("  ▶ Pipeline : Fill-Mask (complétion de mots)")
    print("  " + "-" * 40)
    remplisseur = pipeline("fill-mask", model="bert-base-uncased")

    texte_masque = "Paris is the [MASK] of France."
    resultats = remplisseur(texte_masque)

    print(f"  Texte : \"{texte_masque}\"\n")
    print(f"  Top 5 prédictions :")
    for i, r in enumerate(resultats[:5], 1):
        print(f"    {i}. {r['token_str']:15s} (score: {r['score']:.4f})")
    print()

    # ---- Pipeline 3 : Génération de texte ----
    print("  ▶ Pipeline : Génération de texte (GPT-2)")
    print("  " + "-" * 40)
    generateur = pipeline("text-generation", model="gpt2")

    prompt = "Artificial Intelligence will"
    resultat = generateur(
        prompt,
        max_length=50,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
    )

    print(f"  Prompt : \"{prompt}\"")
    print(f"  Texte généré :")
    print(f"  → {resultat[0]['generated_text']}")
    print()

    # ---- Pipeline 4 : Reconnaissance d'entités nommées (NER) ----
    print("  ▶ Pipeline : NER (Reconnaissance d'Entités Nommées)")
    print("  " + "-" * 40)
    ner = pipeline("ner", aggregation_strategy="simple")

    texte_ner = "Elon Musk founded SpaceX in Hawthorne, California."
    entites = ner(texte_ner)

    print(f"  Texte : \"{texte_ner}\"\n")
    print(f"  Entités détectées :")
    for e in entites:
        print(f"    • {e['word']:20s} → {e['entity_group']:10s} (score: {e['score']:.4f})")
    print()


# ============================================================
# SECTION 4 : Comprendre les embeddings (représentations)
# ============================================================

def explorer_embeddings():
    """
    Montre comment obtenir les embeddings (vecteurs numériques)
    d'un texte avec BERT.

    Un embedding = une représentation numérique d'un mot/phrase
    sous forme de vecteur de nombres réels.
    """
    print("=" * 60)
    print("  Exploration des Embeddings BERT")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Encoder un texte
    texte = "Machine learning is a subset of artificial intelligence."
    inputs = tokenizer(texte, return_tensors="pt", padding=True, truncation=True)

    # Obtenir les embeddings (sans calculer les gradients)
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state = embeddings de chaque token
    # outputs.pooler_output     = embedding de la phrase entière [CLS]
    print(f"\n  Texte : \"{texte}\"\n")
    print(f"  Forme du last_hidden_state : {outputs.last_hidden_state.shape}")
    print(f"    → (batch_size, nb_tokens, dimension_embedding)")
    print(f"    → (1, {outputs.last_hidden_state.shape[1]}, {outputs.last_hidden_state.shape[2]})")
    print()
    print(f"  Forme du pooler_output : {outputs.pooler_output.shape}")
    print(f"    → (batch_size, dimension_embedding)")
    print(f"    → Ceci est l'embedding de la phrase entière")
    print()

    # Calculer la similarité entre deux phrases
    texte_a = "I love programming"
    texte_b = "I enjoy coding"
    texte_c = "The weather is sunny"

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.pooler_output[0]

    emb_a = get_embedding(texte_a)
    emb_b = get_embedding(texte_b)
    emb_c = get_embedding(texte_c)

    # Similarité cosinus
    cos_sim = torch.nn.CosineSimilarity(dim=0)
    sim_ab = cos_sim(emb_a, emb_b).item()
    sim_ac = cos_sim(emb_a, emb_c).item()
    sim_bc = cos_sim(emb_b, emb_c).item()

    print(f"  Similarité cosinus entre phrases :")
    print(f"    \"{texte_a}\" ↔ \"{texte_b}\" : {sim_ab:.4f}")
    print(f"    \"{texte_a}\" ↔ \"{texte_c}\" : {sim_ac:.4f}")
    print(f"    \"{texte_b}\" ↔ \"{texte_c}\" : {sim_bc:.4f}")
    print(f"\n  → Les phrases similaires ont un score plus élevé !")
    print()


# ============================================================
# MAIN - Exécution de toutes les démonstrations
# ============================================================

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("█  PARTIE 1 - ÉTAPE 1 : EXPLORATION DES TRANSFORMERS")
    print("█" * 60 + "\n")

    # 1. Chargement des modèles
    print("\n" + "─" * 60)
    print("  📦 SECTION 1 : Chargement de modèles pré-entraînés")
    print("─" * 60 + "\n")
    model_bert, tok_bert = charger_modele_bert()
    model_gpt2, tok_gpt2 = charger_modele_gpt2()
    model_auto, tok_auto = charger_modele_auto("distilbert-base-uncased")

    # 2. Explorer les tokenizers
    print("\n" + "─" * 60)
    print("  🔤 SECTION 2 : Exploration des Tokenizers")
    print("─" * 60 + "\n")
    explorer_tokenizer()
    comparer_tokenizers()

    # 3. Pipelines simplifiées
    print("\n" + "─" * 60)
    print("  🚀 SECTION 3 : Pipelines Simplifiées")
    print("─" * 60 + "\n")
    demo_pipelines()

    # 4. Embeddings
    print("\n" + "─" * 60)
    print("  🧮 SECTION 4 : Exploration des Embeddings")
    print("─" * 60 + "\n")
    explorer_embeddings()

    print("\n" + "█" * 60)
    print("█  FIN DE L'EXPLORATION DES TRANSFORMERS")
    print("█" * 60 + "\n")
