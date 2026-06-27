"""Built-in seed dataset (offline fallback + tests).

Provides, so everything runs with NO torch and NO network:
* ``DICT_EN_FR`` — an English->French word dictionary for the dictionary MT baseline;
* ``SEED_PAIRS`` — English->French sentence pairs (MT eval/baseline floor);
* ``SEED_DOCS`` — small **structured Markdown documents** with gold French translations
  (headings, lists, code, links, placeholders) so the structure layer + the agent +
  the structure-preservation metric can be exercised end-to-end offline.

Texts are original/synthetic. On Colab the real corpora (opus-100 / news_commentary)
replace the sentence pairs; the structured-document layer is always synthetic.
"""

from __future__ import annotations

from typing import Dict, List

# English -> French sentence pairs (gold)
SEED_PAIRS: List[Dict[str, str]] = [
    {"id": "p01", "src": "Hello, how are you today?", "tgt": "Bonjour, comment allez-vous aujourd'hui ?"},
    {"id": "p02", "src": "The black cat sleeps on the sofa.", "tgt": "Le chat noir dort sur le canape."},
    {"id": "p03", "src": "The meeting starts tomorrow at nine.", "tgt": "La reunion commence demain a neuf heures."},
    {"id": "p04", "src": "The president signed a new agreement.", "tgt": "Le president a signe un nouvel accord."},
    {"id": "p05", "src": "The children play in the park after school.", "tgt": "Les enfants jouent dans le parc apres l'ecole."},
    {"id": "p06", "src": "This company reported good results this year.", "tgt": "Cette entreprise a annonce de bons resultats cette annee."},
    {"id": "p07", "src": "The researchers published a new study.", "tgt": "Les chercheurs ont publie une nouvelle etude."},
    {"id": "p08", "src": "We must reduce pollution in the cities.", "tgt": "Nous devons reduire la pollution dans les villes."},
    {"id": "p09", "src": "The museum is open every day.", "tgt": "Le musee est ouvert tous les jours."},
    {"id": "p10", "src": "She works as a doctor in a hospital.", "tgt": "Elle travaille comme medecin dans un hopital."},
    {"id": "p11", "src": "Thank you very much for your help.", "tgt": "Merci beaucoup pour votre aide."},
    {"id": "p12", "src": "The new phone has a better battery.", "tgt": "Le nouveau telephone a une meilleure batterie."},
    {"id": "p13", "src": "Click the button to open the file.", "tgt": "Cliquez sur le bouton pour ouvrir le fichier."},
    {"id": "p14", "src": "This guide explains how to install the system.", "tgt": "Ce guide explique comment installer le systeme."},
    {"id": "p15", "src": "The report shows the results of the project.", "tgt": "Le rapport montre les resultats du projet."},
    {"id": "p16", "src": "Save your work before you close the application.", "tgt": "Enregistrez votre travail avant de fermer l'application."},
]

# English -> French word dictionary (lowercased; accents stripped to match the baseline).
DICT_EN_FR: Dict[str, str] = {
    "hello": "bonjour", "how": "comment", "are": "allez", "you": "vous", "today": "aujourd'hui",
    "the": "le", "black": "noir", "cat": "chat", "sleeps": "dort", "on": "sur", "sofa": "canape",
    "meeting": "reunion", "starts": "commence", "tomorrow": "demain", "at": "a", "nine": "neuf",
    "president": "president", "signed": "a signe", "a": "un", "new": "nouveau", "agreement": "accord",
    "children": "enfants", "play": "jouent", "in": "dans", "park": "parc", "after": "apres", "school": "ecole",
    "this": "cette", "company": "entreprise", "reported": "a annonce", "good": "bons", "results": "resultats",
    "year": "annee", "researchers": "chercheurs", "published": "ont publie", "study": "etude",
    "we": "nous", "must": "devons", "reduce": "reduire", "pollution": "pollution", "cities": "villes",
    "museum": "musee", "is": "est", "open": "ouvert", "every": "tous", "day": "jours",
    "she": "elle", "works": "travaille", "as": "comme", "doctor": "medecin", "hospital": "hopital",
    "thank": "merci", "thanks": "merci", "very": "tres", "much": "beaucoup", "for": "pour", "your": "votre",
    "help": "aide", "phone": "telephone", "has": "a", "better": "meilleure", "battery": "batterie",
    "click": "cliquez", "button": "bouton", "to": "pour", "file": "fichier", "guide": "guide",
    "explains": "explique", "install": "installer", "system": "systeme", "report": "rapport",
    "shows": "montre", "of": "du", "project": "projet", "save": "enregistrez", "work": "travail",
    "before": "avant", "close": "fermer", "application": "application", "and": "et", "with": "avec",
    "introduction": "introduction", "features": "fonctionnalites", "usage": "utilisation",
    "installation": "installation", "example": "exemple", "note": "note", "warning": "avertissement",
    "run": "executez", "following": "suivant", "command": "commande", "see": "voir", "documentation": "documentation",
    "fast": "rapide", "easy": "facile", "secure": "securise", "supports": "prend en charge", "many": "plusieurs",
    "formats": "formats", "first": "premier", "step": "etape", "second": "deuxieme", "welcome": "bienvenue",
    "user": "utilisateur", "name": "nom", "please": "s'il vous plait", "enter": "entrez",
}

# Small structured Markdown documents with gold French translations (structure preserved).
SEED_DOCS: List[Dict[str, str]] = [
    {
        "id": "d01", "fmt": "markdown",
        "src": (
            "# Welcome\n\n"
            "This guide explains how to install the system.\n\n"
            "## Features\n\n"
            "- Fast and easy\n"
            "- Secure\n"
            "- Supports many formats\n\n"
            "Run the following command:\n\n"
            "```\npip install system\n```\n\n"
            "See the [documentation](https://example.com/docs) for more.\n"
        ),
        "tgt": (
            "# Bienvenue\n\n"
            "Ce guide explique comment installer le systeme.\n\n"
            "## Fonctionnalites\n\n"
            "- Rapide et facile\n"
            "- Securise\n"
            "- Prend en charge plusieurs formats\n\n"
            "Executez la commande suivante :\n\n"
            "```\npip install system\n```\n\n"
            "Voir la [documentation](https://example.com/docs) pour plus.\n"
        ),
    },
    {
        "id": "d02", "fmt": "markdown",
        "src": (
            "## Installation\n\n"
            "1. First step\n"
            "2. Second step\n\n"
            "> Note: save your work before you close the application.\n\n"
            "Welcome `user`, please enter your name.\n"
        ),
        "tgt": (
            "## Installation\n\n"
            "1. Premier etape\n"
            "2. Deuxieme etape\n\n"
            "> Note : enregistrez votre travail avant de fermer l'application.\n\n"
            "Bienvenue `user`, s'il vous plait entrez votre nom.\n"
        ),
    },
]


def pairs() -> List[Dict[str, str]]:
    return [dict(x) for x in SEED_PAIRS]


def dictionary() -> Dict[str, str]:
    return dict(DICT_EN_FR)


def docs() -> List[Dict[str, str]]:
    return [dict(x) for x in SEED_DOCS]


def src_texts() -> List[str]:
    return [p["src"] for p in SEED_PAIRS]


def tgt_texts() -> List[str]:
    return [p["tgt"] for p in SEED_PAIRS]


__all__ = ["SEED_PAIRS", "DICT_EN_FR", "SEED_DOCS", "pairs", "dictionary", "docs",
           "src_texts", "tgt_texts"]
