"""All dataset configurations used in ScandEval."""

from .config import DatasetConfig
from .languages import DA, DE, EN, FO, IS, NB, NL, NN, SV, get_all_languages
from .tasks import COMMON_SENSE, KNOW, LA, NER, QA, SENT, SPEED, SUMM


def get_all_dataset_configs() -> dict[str, DatasetConfig]:
    """Get a mapping of all the dataset configurations.

    Returns:
        A mapping between names of datasets and their configurations.
    """
    return {
        cfg.name: cfg for cfg in globals().values() if isinstance(cfg, DatasetConfig)
    }


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get the dataset configuration for a dataset.

    Args:
        dataset_name:
            The name of the dataset.

    Returns:
        The dataset configuration.

    Raises:
        ValueError:
            If the dataset is not found.
    """
    # Get mapping of all dataset configs
    dataset_configs = get_all_dataset_configs()

    # If there are no matches for the dataset name, raise an error
    if dataset_name not in dataset_configs:
        raise ValueError(f"No dataset config found for dataset {dataset_name}.")

    # Otherwise, return the dataset configuration
    return dataset_configs[dataset_name]


### SENTIMENT DATASETS ###

SWEREC_CONFIG = DatasetConfig(
    name="swerec",
    pretty_name="the truncated version of the Swedish sentiment classification "
    "dataset SweReC",
    huggingface_id="ScandEval/swerec-mini",
    task=SENT,
    languages=[SV],
    prompt_prefix="Följande är recensioner och deras sentiment, som kan vara "
    "'positiv', 'neutral' eller 'negativ'.",
    prompt_template="Recension: {text}\nSentiment: {label}",
    prompt_label_mapping=dict(
        positive="positiv", neutral="neutral", negative="negativ"
    ),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

ANGRY_TWEETS_CONFIG = DatasetConfig(
    name="angry-tweets",
    pretty_name="the truncated version of the Danish sentiment classification "
    "dataset AngryTweets",
    huggingface_id="ScandEval/angry-tweets-mini",
    task=SENT,
    languages=[DA],
    prompt_prefix="Følgende er tweets og deres sentiment, som kan være 'positiv', "
    "'neutral' eller 'negativ'.",
    prompt_template="Tweet: {text}\nSentiment: {label}",
    prompt_label_mapping=dict(
        positive="positiv", neutral="neutral", negative="negativ"
    ),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

NOREC_CONFIG = DatasetConfig(
    name="norec",
    pretty_name="the truncated version of the Norwegian sentiment classification "
    "dataset NoReC",
    huggingface_id="ScandEval/norec-mini",
    task=SENT,
    languages=[NB, NN],
    prompt_prefix="Følgende er anmeldelser og deres sentiment, som kan være 'positiv', "
    "'nøytral' eller 'negativ'.",
    prompt_template="Anmeldelse: {text}\nSentiment: {label}",
    prompt_label_mapping=dict(
        positive="positiv", neutral="nøytral", negative="negativ"
    ),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

# ISREC_CONFIG = DatasetConfig(
#     name="isrec",
#     pretty_name="the truncated version of IsReC",
#     huggingface_id="ScandEval/isrec-mini",  # TODO: Needs to be uploaded
#     task=SENT,
#     languages=[IS],
#     prompt_prefix="Eftirfarandi eru yfirferðir ásamt lyndisgildi þeirra, sem getur "
#     "verið 'jákvætt', 'hlutlaust' eða 'neikvætt'.",
#     prompt_template="Yfirferð: {text}\nLyndi: {label}",
#     prompt_label_mapping=dict(
#         positive="jákvætt", neutral="hlutlaust", negative="neikvætt"
#     ),
#     num_few_shot_examples=12,
#     max_generated_tokens=3,
# )

# FOREC_CONFIG = DatasetConfig(
#     name="forec",
#     pretty_name="the truncated version of FoReC",
#     huggingface_id="ScandEval/forec-mini",  # TODO: Needs to be uploaded
#     task=SENT,
#     languages=[FO],
#     prompt_prefix="Her koma nøkur ummæli og teirra kensluliga sjónarmið, sum kunnu "
#     "vera 'positivur', 'neutralur' ella 'negativur'.",
#     prompt_template="Ummæli: {text}\nKensluligt sjónarmið: {label}",
#     prompt_label_mapping=dict(
#         positive="positivur", neutral="neutralur", negative="negativur"
#     ),
#     num_few_shot_examples=12,
#     max_generated_tokens=3,
# )

SB10K_CONFIG = DatasetConfig(
    name="sb10k",
    pretty_name="the truncated version of the German sentiment classification "
    "dataset SB10k",
    huggingface_id="ScandEval/sb10k-mini",
    task=SENT,
    languages=[DE],
    prompt_prefix="Im Folgenden sind Tweets und ihre Stimmung aufgeführt, die "
    "'positiv', 'neutral' oder 'negativ' sein kann.",
    prompt_template="Tweet: {text}\nStimmungslage: {label}",
    prompt_label_mapping=dict(
        positive="positiv", neutral="neutral", negative="negativ"
    ),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

DUTCH_SOCIAL_CONFIG = DatasetConfig(
    name="dutch-social",
    pretty_name="the truncated version of the Dutch sentiment classification "
    "dataset Dutch Social",
    huggingface_id="ScandEval/dutch-social-mini",
    task=SENT,
    languages=[NL],
    prompt_prefix="Hieronder staan tweets en hun sentiment, dat 'positief', "
    "'neutraal' of 'negatief' kan zijn.",
    prompt_template="Tweet: {text}\nSentiment: {label}",
    prompt_label_mapping=dict(
        positive="positief", neutral="neutraal", negative="negatief"
    ),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

SST5_CONFIG = DatasetConfig(
    name="sst5",
    pretty_name="the truncated version of the English sentiment classification "
    "dataset SST5",
    huggingface_id="ScandEval/sst5-mini",
    task=SENT,
    languages=[EN],
    prompt_prefix="The following are texts are their sentiment, which can be "
    "'positive', 'neutral' or 'negative'.",
    prompt_template="Text: {text}\nSentiment: {label}",
    prompt_label_mapping=dict(
        positive="positive", neutral="neutral", negative="negative"
    ),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

# TODO: Icelandic Sentiment Classification
# TODO: Faroese Sentiment Classification


### NAMED ENTITY RECOGNITION DATASETS ###

SUC3_CONFIG = DatasetConfig(
    name="suc3",
    pretty_name="the truncated version of the Swedish named entity recognition "
    "dataset SUC 3.0",
    huggingface_id="ScandEval/suc3-mini",
    task=NER,
    languages=[SV],
    prompt_prefix="Följande är meningar och JSON-ordböcker med de namngivna enheter "
    "som förekommer i den givna meningen.",
    prompt_template="Mening: {text}\nNamngivna entiteter: {label}",
    prompt_label_mapping={
        "b-per": "person",
        "i-per": "person",
        "b-loc": "plats",
        "i-loc": "plats",
        "b-org": "organisation",
        "i-org": "organisation",
        "b-misc": "diverse",
        "i-misc": "diverse",
    },
    num_few_shot_examples=8,
    max_generated_tokens=128,
)

DANSK_CONFIG = DatasetConfig(
    name="dansk",
    pretty_name="the truncated version of the Danish named entity recognition "
    "dataset DANSK",
    huggingface_id="ScandEval/dansk-mini",
    task=NER,
    languages=[DA],
    prompt_prefix="Følgende er sætninger og JSON-ordbøger med de navngivne enheder, "
    "som forekommer i den givne sætning.",
    prompt_template="Sætning: {text}\nNavngivne enheder: {label}",
    prompt_label_mapping={
        "b-per": "person",
        "i-per": "person",
        "b-loc": "sted",
        "i-loc": "sted",
        "b-org": "organisation",
        "i-org": "organisation",
        "b-misc": "diverse",
        "i-misc": "diverse",
    },
    num_few_shot_examples=8,
    max_generated_tokens=128,
)

NORNE_NB_CONFIG = DatasetConfig(
    name="norne-nb",
    pretty_name="the truncated version of the Bokmål part of the Norwegian named "
    "entity recognition dataset NorNE",
    huggingface_id="ScandEval/norne-nb-mini",
    task=NER,
    languages=[NB],
    prompt_prefix="Følgende er fraser og JSON-ordbøker med de navngitte enhetene "
    "som forekommer i den gitte frasen.",
    prompt_template="Frase: {text}\nNavngitte enheter: {label}",
    prompt_label_mapping={
        "b-per": "person",
        "i-per": "person",
        "b-loc": "sted",
        "i-loc": "sted",
        "b-org": "organisasjon",
        "i-org": "organisasjon",
        "b-misc": "diverse",
        "i-misc": "diverse",
    },
    num_few_shot_examples=8,
    max_generated_tokens=128,
)

NORNE_NN_CONFIG = DatasetConfig(
    name="norne-nn",
    pretty_name="the truncated version of the Nynorsk part of the Norwegian named "
    "entity recognition dataset NorNE",
    huggingface_id="ScandEval/norne-nn-mini",
    task=NER,
    languages=[NN],
    prompt_prefix="Følgende er fraser og JSON-ordbøker med de navngitte enhetene "
    "som forekommer i den gitte frasen.",
    prompt_template="Frase: {text}\nNavngitte enheter: {label}",
    prompt_label_mapping={
        "b-per": "person",
        "i-per": "person",
        "b-loc": "sted",
        "i-loc": "sted",
        "b-org": "organisasjon",
        "i-org": "organisasjon",
        "b-misc": "diverse",
        "i-misc": "diverse",
    },
    num_few_shot_examples=8,
    max_generated_tokens=128,
)

MIM_GOLD_NER_CONFIG = DatasetConfig(
    name="mim-gold-ner",
    pretty_name="the truncated version of the Icelandic named entity recognition "
    "dataset MIM-GOLD-NER",
    huggingface_id="ScandEval/mim-gold-ner-mini",
    task=NER,
    languages=[IS],
    prompt_prefix="Eftirfarandi eru setningar ásamt JSON lyklum með nefndum einingum "
    "sem koma fyrir í setningunum.",
    prompt_template="Setning: {text}\nNefndar einingar: {label}",
    prompt_label_mapping={
        "b-per": "einstaklingur",
        "i-per": "einstaklingur",
        "b-loc": "staðsetning",
        "i-loc": "staðsetning",
        "b-org": "stofnun",
        "i-org": "stofnun",
        "b-misc": "ýmislegt",
        "i-misc": "ýmislegt",
    },
    num_few_shot_examples=8,
    max_generated_tokens=128,
)

FONE_CONFIG = DatasetConfig(
    name="fone",
    pretty_name="the truncated version of the Faroese named entity recognition "
    "dataset FoNE",
    huggingface_id="ScandEval/fone-mini",
    task=NER,
    languages=[FO],
    prompt_prefix="Her eru nakrir setningar og nakrar JSON orðabøkur við nevndar "
    "eindir, sum eru í setningunum.",
    prompt_template="Setningur: {text}\nNevndar eindir: {label}",
    prompt_label_mapping={
        "b-per": "persónur",
        "i-per": "persónur",
        "b-loc": "staður",
        "i-loc": "staður",
        "b-org": "felagsskapur",
        "i-org": "felagsskapur",
        "b-misc": "ymiskt",
        "i-misc": "ymiskt",
    },
    num_few_shot_examples=8,
    max_generated_tokens=128,
)

GERMEVAL_CONFIG = DatasetConfig(
    name="germeval",
    pretty_name="the truncated version of the German named entity recognition "
    "dataset GermEval",
    huggingface_id="ScandEval/germeval-mini",
    task=NER,
    languages=[DE],
    prompt_prefix="Es folgen Sätze und JSON-Wörterbücher mit den benannten "
    "Entitäten, die in der angegebenen Phrase vorkommen.",
    prompt_template="Satz: {text}\nBenannte Entitäten: {label}",
    prompt_label_mapping={
        "b-per": "person",
        "i-per": "person",
        "b-loc": "ort",
        "i-loc": "ort",
        "b-org": "organisation",
        "i-org": "organisation",
        "b-misc": "verschiedenes",
        "i-misc": "verschiedenes",
    },
    num_few_shot_examples=8,
    max_generated_tokens=128,
)

CONLL_NL_CONFIG = DatasetConfig(
    name="conll-nl",
    pretty_name="the Dutch part of the truncated version of the named entity "
    "recognition dataset CoNLL 2002",
    huggingface_id="ScandEval/conll-nl-mini",
    task=NER,
    languages=[NL],
    prompt_prefix="Hieronder staan zinnen en JSON woordenboeken met de genoemde "
    "entiteiten die voorkomen in de gegeven zin.",
    prompt_template="Zin: {text}\nGenoemde entiteiten: {label}",
    prompt_label_mapping={
        "b-per": "persoon",
        "i-per": "persoon",
        "b-loc": "locatie",
        "i-loc": "locatie",
        "b-org": "organisatie",
        "i-org": "organisatie",
        "b-misc": "diversen",
        "i-misc": "diversen",
    },
    num_few_shot_examples=8,
    max_generated_tokens=128,
)

CONLL_EN_CONFIG = DatasetConfig(
    name="conll-en",
    pretty_name="the truncated version of the English named entity recognition "
    "dataset CoNLL 2003",
    huggingface_id="ScandEval/conll-en-mini",
    task=NER,
    languages=[EN],
    prompt_prefix="Below are sentences and JSON dictionaries with the named "
    "entities that occur in the given sentence.",
    prompt_template="Sentence: {text}\nNamed entities: {label}",
    prompt_label_mapping={
        "b-per": "person",
        "i-per": "person",
        "b-loc": "location",
        "i-loc": "location",
        "b-org": "organization",
        "i-org": "organization",
        "b-misc": "miscellaneous",
        "i-misc": "miscellaneous",
    },
    num_few_shot_examples=8,
    max_generated_tokens=128,
)


### LINGUISTIC ACCEPTABILITY DATASETS ###

SCALA_SV_CONFIG = DatasetConfig(
    name="scala-sv",
    pretty_name="The Swedish part of the linguistic acceptability dataset ScaLA",
    huggingface_id="ScandEval/scala-sv",
    task=LA,
    languages=[SV],
    prompt_prefix="Följande är meningar och huruvida de är grammatiskt korrekta.",
    prompt_template="Mening: {text}\nGrammatisk korrekt: {label}",
    prompt_label_mapping=dict(correct="ja", incorrect="nej"),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

SCALA_DA_CONFIG = DatasetConfig(
    name="scala-da",
    pretty_name="the Danish part of the linguistic acceptability dataset ScaLA",
    huggingface_id="ScandEval/scala-da",
    task=LA,
    languages=[DA],
    prompt_prefix="Følgende er sætninger og om de er grammatisk korrekte.",
    prompt_template="Sætning: {text}\nGrammatisk korrekt: {label}",
    prompt_label_mapping=dict(correct="ja", incorrect="nej"),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

SCALA_NB_CONFIG = DatasetConfig(
    name="scala-nb",
    pretty_name="the Bokmål part of the linguistic acceptability dataset ScaLA",
    huggingface_id="ScandEval/scala-nb",
    task=LA,
    languages=[NB],
    prompt_prefix="Følgende er setninger og hvorvidt de er grammatisk korrekte.",
    prompt_template="Setning: {text}\nGrammatisk korrekt: {label}",
    prompt_label_mapping=dict(correct="ja", incorrect="nei"),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

SCALA_NN_CONFIG = DatasetConfig(
    name="scala-nn",
    pretty_name="the Nynorsk part of the linguistic acceptability dataset ScaLA",
    huggingface_id="ScandEval/scala-nn",
    task=LA,
    languages=[NN],
    prompt_prefix="Følgende er setninger og hvorvidt de er grammatisk korrekte.",
    prompt_template="Setning: {text}\nGrammatisk korrekt: {label}",
    prompt_label_mapping=dict(correct="ja", incorrect="nei"),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

SCALA_IS_CONFIG = DatasetConfig(
    name="scala-is",
    pretty_name="the Icelandic part of the linguistic acceptability dataset ScaLA",
    huggingface_id="ScandEval/scala-is",
    task=LA,
    languages=[IS],
    prompt_prefix="Eftirfarandi eru setningar og hvort þær eru málfræðilega réttar.",
    prompt_template="Setning: {text}\nMálfræðilega rétt: {label}",
    prompt_label_mapping=dict(correct="já", incorrect="nei"),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)


SCALA_FO_CONFIG = DatasetConfig(
    name="scala-fo",
    pretty_name="the Faroese part of the linguistic acceptability dataset ScaLA",
    huggingface_id="ScandEval/scala-fo",
    task=LA,
    languages=[FO],
    prompt_prefix="Hetta eru nakrir setningar og um teir eru mállæruliga rættir.",
    prompt_template="Setningur: {text}\nMállæruliga rættur: {label}",
    prompt_label_mapping=dict(correct="ja", incorrect="nei"),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

SCALA_DE_CONFIG = DatasetConfig(
    name="scala-de",
    pretty_name="the German part of the linguistic acceptability dataset ScaLA",
    huggingface_id="ScandEval/scala-de",
    task=LA,
    languages=[DE],
    prompt_prefix="Die folgenden Sätze und ob sie grammatikalisch korrekt sind.",
    prompt_template="Satz: {text}\nGrammatikalisch richtig: {label}",
    prompt_label_mapping=dict(correct="ja", incorrect="nein"),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

SCALA_NL_CONFIG = DatasetConfig(
    name="scala-nl",
    pretty_name="the Dutch part of the linguistic acceptability dataset ScaLA",
    huggingface_id="ScandEval/scala-nl",
    task=LA,
    languages=[NL],
    prompt_prefix="Hieronder staan zinnen en of ze grammaticaal correct zijn.",
    prompt_template="Zin: {text}\nGrammaticaal correct: {label}",
    prompt_label_mapping=dict(correct="ja", incorrect="nee"),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)

SCALA_EN_CONFIG = DatasetConfig(
    name="scala-en",
    pretty_name="the English part of the linguistic acceptability dataset ScaLA",
    huggingface_id="ScandEval/scala-en",
    task=LA,
    languages=[EN],
    prompt_prefix="The following are sentences and whether they are grammatically "
    "correct.",
    prompt_template="Sentence: {text}\nGrammatically correct: {label}",
    prompt_label_mapping=dict(correct="yes", incorrect="no"),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)


### EXTRACTIVE QUESTION ANSWERING DATASETS ###

SCANDIQA_DA_CONFIG = DatasetConfig(
    name="scandiqa-da",
    pretty_name="the Danish part of the truncated version of the question answering "
    "dataset ScandiQA",
    huggingface_id="ScandEval/scandiqa-da-mini",
    task=QA,
    languages=[DA],
    prompt_prefix="Følgende er tekster med tilhørende spørgsmål og svar.",
    prompt_template="Tekst: {text}\nSpørgsmål: {question}\nSvar med maks. 3 ord: "
    "{label}",
    num_few_shot_examples=4,
    max_generated_tokens=32,
)

NORQUAD_CONFIG = DatasetConfig(
    name="norquad",
    pretty_name="the truncated version of the Norwegian question answering "
    "dataset NorQuAD",
    huggingface_id="ScandEval/norquad-mini",
    task=QA,
    languages=[NB, NN],
    prompt_prefix="Her følger tekster med tilhørende spørsmål og svar.",
    prompt_template="Tekst: {text}\nSpørsmål: {question}\nSvar på maks 3 ord: {label}",
    num_few_shot_examples=2,
    max_generated_tokens=32,
)

SCANDIQA_SV_CONFIG = DatasetConfig(
    name="scandiqa-sv",
    pretty_name="the Swedish part of the truncated version of the question answering "
    "dataset ScandiQA",
    huggingface_id="ScandEval/scandiqa-sv-mini",
    task=QA,
    languages=[SV],
    prompt_prefix="Nedan följer texter med tillhörande frågor och svar.",
    prompt_template="Text: {text}\nFråga: {question}\nSvar på max 3 ord: {label}",
    num_few_shot_examples=4,
    max_generated_tokens=32,
)

NQII_CONFIG = DatasetConfig(
    name="nqii",
    pretty_name="the truncated version of the Icelandic question answering dataset "
    "Natural Questions in Icelandic",
    huggingface_id="ScandEval/nqii-mini",
    task=QA,
    languages=[IS],
    prompt_prefix="Eftirfarandi eru textar með tilheyrandi spurningum og svörum.",
    prompt_template="Texti: {text}\nSpurning: {question}\nSvaraðu með að hámarki 3 "
    "orðum: {label}",
    num_few_shot_examples=4,
    max_generated_tokens=32,
)

# FOQA_CONFIG = DatasetConfig(
#     name="foqa",
#     pretty_name="Faroese Question Answering",
#     huggingface_id="ScandEval/foqa-mini",  # TODO: Needs to be uploaded
#     task=QA,
#     languages=[FO],
#     prompt_template="{text}\nSpurningur: {question}\nSvara við í mesta lagi trimum "
#     "orðum: {label}",
#     num_few_shot_examples=4,
#     max_generated_tokens=32,
# )

GERMANQUAD_CONFIG = DatasetConfig(
    name="germanquad",
    pretty_name="the truncated version of the German question answering dataset "
    "GermanQuAD",
    huggingface_id="ScandEval/germanquad-mini",
    task=QA,
    languages=[DE],
    prompt_prefix="Im Folgenden finden Sie Texte mit den dazugehörigen Fragen und "
    "Antworten.",
    prompt_template="Text: {text}\nFragen: {question}\nFragen Antwort in maximal 3 "
    "Wörtern: {label}",
    num_few_shot_examples=4,
    max_generated_tokens=32,
)

SQUAD_CONFIG = DatasetConfig(
    name="squad",
    pretty_name="the truncated version of the English question answering "
    "dataset SQuAD",
    huggingface_id="ScandEval/squad-mini",
    task=QA,
    languages=[EN],
    prompt_prefix="The following are texts with accompanying questions and answers.",
    prompt_template="Text: {text}\nQuestion: {question}\nAnswer in max 3 words: "
    "{label}",
    num_few_shot_examples=4,
    max_generated_tokens=32,
)

SQUAD_NL_CONFIG = DatasetConfig(
    name="squad-nl",
    pretty_name="the truncated version of the Dutch question answering dataset "
    "SQuAD-nl, translated from the English SQuAD dataset",
    huggingface_id="ScandEval/squad-nl-mini",
    task=QA,
    languages=[NL],
    prompt_prefix="Hieronder volgen teksten met bijbehorende vragen en antwoorden.",
    prompt_template="Tekst: {text}\nVraag: {question}\nAntwoord in max 3 woorden: "
    "{label}",
    num_few_shot_examples=4,
    max_generated_tokens=32,
)

# TODO: Faroese Question Answering


### SUMMARIZATION DATASETS ###

NORDJYLLAND_NEWS_CONFIG = DatasetConfig(
    name="nordjylland-news",
    pretty_name="the truncated version of the Danish summarisation dataset "
    "Nordjylland News",
    huggingface_id="ScandEval/nordjylland-news-mini",
    task=SUMM,
    languages=[DA],
    prompt_prefix="Følgende er nyhedsartikler med tilhørende resuméer.",
    prompt_template="Nyhedsartikel: {text}\nResumé: {target_text}",
    num_few_shot_examples=1,
    max_generated_tokens=128,
)

MLSUM_CONFIG = DatasetConfig(
    name="mlsum",
    pretty_name="the truncated version of the German summarisation dataset MLSum",
    huggingface_id="ScandEval/mlsum-mini",
    task=SUMM,
    languages=[DE],
    prompt_prefix="Im Folgenden finden Sie Nachrichtenartikel mit den dazugehörigen "
    "Zusammenfassungen.",
    prompt_template="Nachrichtenartikel: {text}\nZusammenfassung: {target_text}",
    num_few_shot_examples=1,
    max_generated_tokens=256,
)

RRN_CONFIG = DatasetConfig(
    name="rrn",
    pretty_name="the truncated version of the Icelandic summarisation dataset "
    "RÚV Radio News",
    huggingface_id="ScandEval/rrn-mini",
    task=SUMM,
    languages=[IS],
    prompt_prefix="Eftirfarandi eru fréttagreinar með tilheyrandi samantektum.",
    prompt_template="Fréttagrein: {text}\nSamantekt: {target_text}",
    num_few_shot_examples=1,
    max_generated_tokens=256,
)

NO_SAMMENDRAG_CONFIG = DatasetConfig(
    name="no-sammendrag",
    pretty_name="the truncated version of the Norwegian summarisation dataset "
    "Norske Sammendrag",
    huggingface_id="ScandEval/no-sammendrag-mini",
    task=SUMM,
    languages=[NB, NN],
    prompt_prefix="Her følger nyhetsartikler med tilhørende sammendrag.",
    prompt_template="Nyhetsartikkel: {text}\nSammendrag: {target_text}",
    num_few_shot_examples=1,
    max_generated_tokens=256,
)

WIKI_LINGUA_NL_CONFIG = DatasetConfig(
    name="wiki-lingua-nl",
    pretty_name="the Dutch part of the truncated version of the summarisation dataset "
    " WikiLingua",
    huggingface_id="ScandEval/wiki-lingua-nl-mini",
    task=SUMM,
    languages=[NL],
    prompt_prefix="Hieronder volgen artikelen met bijbehorende samenvattingen.",
    prompt_template="Artikel: {text}\nSamenvatting: {target_text}",
    num_few_shot_examples=1,
    max_generated_tokens=256,
)

SWEDN_CONFIG = DatasetConfig(
    name="swedn",
    pretty_name="the truncated version of the Swedish summarisation dataset SweDN",
    huggingface_id="ScandEval/swedn-mini",
    task=SUMM,
    languages=[SV],
    prompt_prefix="Nedan följer artiklar med tillhörande sammanfattningar.",
    prompt_template="Artikel: {text}\nSammanfattning: {target_text}",
    num_few_shot_examples=1,
    max_generated_tokens=256,
)

CNN_DAILYMAIL_CONFIG = DatasetConfig(
    name="cnn-dailymail",
    pretty_name="the truncated version of the English summarisation dataset "
    "CNN-DailyMail",
    huggingface_id="ScandEval/cnn-dailymail-mini",
    task=SUMM,
    languages=[EN],
    prompt_prefix="The following are articles with accompanying summaries.",
    prompt_template="News article: {text}\nSummary: {target_text}",
    num_few_shot_examples=1,
    max_generated_tokens=256,
)

# TODO: Faroese summarization


### KNOWLEDGE DATASETS ###

DANSKE_TALEMAADER_CONFIG = DatasetConfig(
    name="danske-talemaader",
    pretty_name="the truncated version of the Danish knowledge dataset Danske "
    "Talemåder",
    huggingface_id="ScandEval/danske-talemaader-mini",
    task=KNOW,
    languages=[DA],
    prompt_prefix="Følgende er multiple choice spørgsmål (med svar).",
    prompt_template="Hvad er betydningen af følgende talemåde: {text}\nSvar: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

DANISH_CITIZEN_TESTS_CONFIG = DatasetConfig(
    name="danish-citizen-tests",
    pretty_name="the Danish knowledge dataset Danish Citizen Tests",
    huggingface_id="ScandEval/danish-citizen-tests",
    task=KNOW,
    languages=[DA],
    prompt_prefix="Følgende er multiple choice spørgsmål (med svar).",
    prompt_template="Spørgsmål: {text}\nSvar: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

MMLU_NO_CONFIG = DatasetConfig(
    name="mmlu-no",
    pretty_name="the truncated version of the Norwegian knowledge dataset MMLU-no, "
    "translated from the English MMLU dataset",
    huggingface_id="ScandEval/mmlu-no-mini",
    task=KNOW,
    languages=[NB, NN],
    prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
    prompt_template="Spørsmål: {text}\nSvar: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

MMLU_SV_CONFIG = DatasetConfig(
    name="mmlu-sv",
    pretty_name="the truncated version of the Swedish knowledge dataset MMLU-sv, "
    "translated from the English MMLU dataset",
    huggingface_id="ScandEval/mmlu-sv-mini",
    task=KNOW,
    languages=[SV],
    prompt_prefix="Följande är flervalsfrågor (med svar).",
    prompt_template="Fråga: {text}\nSvar: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

MMLU_IS_CONFIG = DatasetConfig(
    name="mmlu-is",
    pretty_name="the truncated version of the Icelandic knowledge dataset MMLU-is, "
    "translated from the English MMLU dataset",
    huggingface_id="ScandEval/mmlu-is-mini",
    task=KNOW,
    languages=[IS],
    prompt_prefix="Eftirfarandi eru fjölvalsspurningar (með svörum).",
    prompt_template="Spurningar: {text}\nSvara: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

MMLU_DE_CONFIG = DatasetConfig(
    name="mmlu-de",
    pretty_name="the truncated version of the German knowledge dataset MMLU-de, "
    "translated from the English MMLU dataset",
    huggingface_id="ScandEval/mmlu-de-mini",
    task=KNOW,
    languages=[DE],
    prompt_prefix="Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).",
    prompt_template="Frage: {text}\nAntwort: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

MMLU_NL_CONFIG = DatasetConfig(
    name="mmlu-nl",
    pretty_name="the truncated version of the Dutch knowledge dataset MMLU-nl, "
    "translated from the English MMLU dataset",
    huggingface_id="ScandEval/mmlu-nl-mini",
    task=KNOW,
    languages=[NL],
    prompt_prefix="Hieronder staan meerkeuzevragen (met antwoorden).",
    prompt_template="Vraag: {text}\nAntwoord: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

MMLU_CONFIG = DatasetConfig(
    name="mmlu",
    pretty_name="the truncated version of the English knowledge dataset MMLU",
    huggingface_id="ScandEval/mmlu-mini",
    task=KNOW,
    languages=[EN],
    prompt_prefix="The following are multiple choice questions (with answers).",
    prompt_template="Question: {text}\nAnswer: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

# TODO: Faroese knowledge


### COMMON SENSE REASONING DATASETS ###

HELLASWAG_DA_CONFIG = DatasetConfig(
    name="hellaswag-da",
    pretty_name="the truncated version of the Danish common-sense reasoning dataset "
    "HellaSwag-da, translated from the English HellaSwag dataset",
    huggingface_id="ScandEval/hellaswag-da-mini",
    task=COMMON_SENSE,
    languages=[DA],
    prompt_prefix="Følgende er multiple choice spørgsmål (med svar).",
    prompt_template="Spørgsmål: {text}\nSvar: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

HELLASWAG_NO_CONFIG = DatasetConfig(
    name="hellaswag-no",
    pretty_name="the truncated version of the Norwegian common-sense reasoning dataset "
    "HellaSwag-no, translated from the English HellaSwag dataset",
    huggingface_id="ScandEval/hellaswag-no-mini",
    task=COMMON_SENSE,
    languages=[NB, NN],
    prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
    prompt_template="Spørsmål: {text}\nSvar: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

HELLASWAG_SV_CONFIG = DatasetConfig(
    name="hellaswag-sv",
    pretty_name="the truncated version of the Swedish common-sense reasoning dataset "
    "HellaSwag-sv, translated from the English HellaSwag dataset",
    huggingface_id="ScandEval/hellaswag-sv-mini",
    task=COMMON_SENSE,
    languages=[SV],
    prompt_prefix="Följande är flervalsfrågor (med svar).",
    prompt_template="Fråga: {text}\nSvar: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

HELLASWAG_IS_CONFIG = DatasetConfig(
    name="hellaswag-is",
    pretty_name="the truncated version of the Icelandic common-sense reasoning dataset "
    "HellaSwag-is, translated from the English HellaSwag dataset",
    huggingface_id="ScandEval/hellaswag-is-mini",
    task=COMMON_SENSE,
    languages=[IS],
    prompt_prefix="Eftirfarandi eru fjölvalsspurningar (með svörum).",
    prompt_template="Spurningar: {text}\nSvara: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

HELLASWAG_DE_CONFIG = DatasetConfig(
    name="hellaswag-de",
    pretty_name="the truncated version of the German common-sense reasoning dataset "
    "HellaSwag-de, translated from the English HellaSwag dataset",
    huggingface_id="ScandEval/hellaswag-de-mini",
    task=COMMON_SENSE,
    languages=[DE],
    prompt_prefix="Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).",
    prompt_template="Frage: {text}\nAntwort: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

HELLASWAG_NL_CONFIG = DatasetConfig(
    name="hellaswag-nl",
    pretty_name="the truncated version of the Dutch common-sense reasoning dataset "
    "HellaSwag-nl, translated from the English HellaSwag dataset",
    huggingface_id="ScandEval/hellaswag-nl-mini",
    task=COMMON_SENSE,
    languages=[NL],
    prompt_prefix="Hieronder staan meerkeuzevragen (met antwoorden).",
    prompt_template="Vraag: {text}\nAntwoord: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

HELLASWAG_CONFIG = DatasetConfig(
    name="hellaswag",
    pretty_name="the truncated version of the English common-sense reasoning "
    "dataset HellaSwag",
    huggingface_id="ScandEval/hellaswag-mini",
    task=COMMON_SENSE,
    languages=[EN],
    prompt_prefix="The following are multiple choice questions (with answers).",
    prompt_template="Question: {text}\nAnswer: {label}",
    prompt_label_mapping=dict(a="a", b="b", c="c", d="d"),
    num_few_shot_examples=5,
    max_generated_tokens=3,
)

# TODO: Faroese common sense reasoning


### SPEED ESTIMATION DATASETS ###

SPEED_CONFIG = DatasetConfig(
    name="speed",
    pretty_name="the speed estimation benchmark",
    huggingface_id="",
    task=SPEED,
    languages=list(get_all_languages().values()),
    prompt_template="",
    max_generated_tokens=1,
)
