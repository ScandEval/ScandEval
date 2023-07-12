"""All dataset configurations used in ScandEval."""

from .config import DatasetConfig
from .dataset_tasks import LA, NER, QA, SENT, SPEED
from .languages import DA, FO, IS, NB, NN, SV, get_all_languages


def get_all_dataset_configs() -> dict[str, DatasetConfig]:
    """Get a mapping of all the dataset configurations.

    Returns:
        dict:
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
        DatasetConfig:
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


SWEREC_CONFIG = DatasetConfig(
    name="swerec",
    pretty_name="the truncated version of SweReC",
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
    pretty_name="the truncated version of AngryTweets",
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
    pretty_name="the truncated version of NoReC",
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


SUC3_CONFIG = DatasetConfig(
    name="suc3",
    pretty_name="the truncated version of SUC 3.0",
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


DANE_CONFIG = DatasetConfig(
    name="dane",
    pretty_name="the truncated version of DaNE",
    huggingface_id="ScandEval/dane-mini",
    task=NER,
    languages=[DA],
    prompt_prefix="Det følgende er dokumenter og deres navngivne enheder, som kan være "
    "'person', 'sted', 'organisation' og 'diverse':",
    prompt_template="Dokument: {text}\nNavngivne enheder: {label}",
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
    pretty_name="the truncated version of the Bokmål part of NorNE",
    huggingface_id="ScandEval/norne-nb-mini",
    task=NER,
    languages=[NB],
    prompt_prefix="Følgende er dokumenter og deres navngitte enheter, som kan være "
    "'person', 'sted', 'organisasjon' og 'diverse':",
    prompt_template="Dokument: {text}\nNavngitte enheter: {label}",
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
    pretty_name="the truncated version of the Nynorsk part of NorNE",
    huggingface_id="ScandEval/norne-nn-mini",
    task=NER,
    languages=[NN],
    prompt_prefix="Følgende er dokumenter og deres navngitte enheter, som kan være "
    "'person', 'sted', 'organisasjon' og 'diverse':",
    prompt_template="Dokument: {text}\nNavngitte enheter: {label}",
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
    pretty_name="the truncated version of MIM-GOLD-NER",
    huggingface_id="ScandEval/mim-gold-ner-mini",
    task=NER,
    languages=[IS],
    prompt_prefix="Eftirfarandi eru skjöl og nafngreindir aðilar þeirra, sem geta "
    "verið 'persóna', 'staðsetning', 'stofnun' og 'ýmislegt':",
    prompt_template="Dokument: {text}\nNafngreindir aðilar: {label}",
    prompt_label_mapping={
        "b-per": "persóna",
        "i-per": "persóna",
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


WIKIANN_FO_CONFIG = DatasetConfig(
    name="wikiann-fo",
    pretty_name="the truncated version of the Faroese part of WikiANN",
    huggingface_id="ScandEval/wikiann-fo-mini",
    task=NER,
    languages=[FO],
    prompt_prefix="",  # TODO
    prompt_template="",  # TODO
    prompt_label_mapping=dict(),
    num_few_shot_examples=8,
    max_generated_tokens=128,
)


FONE_CONFIG = DatasetConfig(
    name="fone",
    pretty_name="the truncated version of the FoNE dataset",
    huggingface_id="ScandEval/fone-mini",  # TODO: Needs to be uploaded
    task=NER,
    languages=[FO],
    prompt_prefix="",  # TODO
    prompt_template="",  # TODO
    prompt_label_mapping=dict(),
    num_few_shot_examples=8,
    max_generated_tokens=128,
)


SCALA_SV_CONFIG = DatasetConfig(
    name="scala-sv",
    pretty_name="The Swedish part of ScaLA",
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
    pretty_name="the Danish part of ScaLA",
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
    pretty_name="the Bokmål part of ScaLA",
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
    pretty_name="the Nynorsk part of ScaLA",
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
    pretty_name="the Icelandic part of ScaLA",
    huggingface_id="ScandEval/scala-is",
    task=LA,
    languages=[IS],
    prompt_prefix="Eftirfarandi eru setningar og hvort þær séu málfræðilega réttar.",
    prompt_template="Setning: {text}\nMálfræðilega rétt: {label}",
    prompt_label_mapping=dict(correct="já", incorrect="nei"),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)


SCALA_FO_CONFIG = DatasetConfig(
    name="scala-fo",
    pretty_name="the Faroese part of ScaLA",
    huggingface_id="ScandEval/scala-fo",
    task=LA,
    languages=[FO],
    prompt_prefix="Etta eru setningar og um teir eru grammatiskt rætt.",
    prompt_template="Setning: {text}\nGrammatiskt rætt: {label}",
    prompt_label_mapping=dict(correct="ja", incorrect="nei"),
    num_few_shot_examples=12,
    max_generated_tokens=3,
)


SCANDIQA_DA_CONFIG = DatasetConfig(
    name="scandiqa-da",
    pretty_name="the Danish part of the truncated version of ScandiQA",
    huggingface_id="ScandEval/scandiqa-da-mini",
    task=QA,
    languages=[DA],
    prompt_template="{text}\nSpørgsmål: {question}\nSvar med maks. 3 ord: {label}",
    num_few_shot_examples=4,
    max_generated_tokens=32,
)


SCANDIQA_NO_CONFIG = DatasetConfig(
    name="scandiqa-no",
    pretty_name="the Norwegian part of truncated version of ScandiQA",
    huggingface_id="ScandEval/scandiqa-no-mini",
    task=QA,
    languages=[NB, NN],
    prompt_template="{text}\nSpørsmål: {question}\nSvar på maks 3 ord: {label}",
    num_few_shot_examples=4,
    max_generated_tokens=32,
)


SCANDIQA_SV_CONFIG = DatasetConfig(
    name="scandiqa-sv",
    pretty_name="the Swedish part of the truncated version of ScandiQA",
    huggingface_id="ScandEval/scandiqa-sv-mini",
    task=QA,
    languages=[SV],
    prompt_template="{text}\nFråga: {question}\nSvar på max 3 ord: {label}",
    num_few_shot_examples=4,
    max_generated_tokens=32,
)


NQII_CONFIG = DatasetConfig(
    name="nqii",
    pretty_name="Natural Questions in Icelandic",
    huggingface_id="ScandEval/nqii-mini",  # TODO: Needs to be uploaded
    task=QA,
    languages=[IS],
    prompt_template="{text}\nSpurning: {question}\nSvar að hámarki 3 orð: {label}",
    num_few_shot_examples=4,
    max_generated_tokens=32,
)


SPEED_CONFIG = DatasetConfig(
    name="speed",
    pretty_name="the speed estimation benchmark",
    huggingface_id="",
    task=SPEED,
    languages=list(get_all_languages().values()),
    prompt_template="",
    max_generated_tokens=1,
)
