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
        dataset_name (str):
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
    prompt_template="{text}",  # TODO
)


ANGRY_TWEETS_CONFIG = DatasetConfig(
    name="angry-tweets",
    pretty_name="the truncated version of AngryTweets",
    huggingface_id="ScandEval/angry-tweets-mini",
    task=SENT,
    languages=[DA],
    prompt_template="""
Tweet: Et stort tillykke til @USER og vinderne af årets Cavlingpris 💪 [LINK]
Thoughts: The tweet is congratulating someone and uses the emoji 💪 to emphasise that they did hard work.
Label: positive

Tweet: @USER En international bilateral aftale er f.eks. en aftale mellem 2 lande i modsætning til en indgået mellem mange lande.
Thoughts: The tweet is explaining the meaning of the word "bilateral" to another user, in a polite and informative manner.
Label: neutral

Tweet: Hvorfor er det, at vi ikke længere må se TV-billederne af gengivelserne, når der er VAR? Det er da grænseløst irriterende #CLDK
Thoughts: The tweet is expressing frustration about not being able to see the TV images when the VAR is on.
Label: negative

Tweet: {text}
Thoughts: """,
)


NOREC_CONFIG = DatasetConfig(
    name="norec",
    pretty_name="the truncated version of NoReC",
    huggingface_id="ScandEval/norec-mini",
    task=SENT,
    languages=[NB, NN],
    prompt_template="{text}",  # TODO
)


SUC3_CONFIG = DatasetConfig(
    name="suc3",
    pretty_name="the truncated version of SUC 3.0",
    huggingface_id="ScandEval/suc3-mini",
    task=NER,
    languages=[SV],
    prompt_template="{text}",  # TODO
)


DANE_CONFIG = DatasetConfig(
    name="dane",
    pretty_name="the truncated version of DaNE",
    huggingface_id="ScandEval/dane-mini",
    task=NER,
    languages=[DA],
    prompt_template="{text}",  # TODO
)


NORNE_NB_CONFIG = DatasetConfig(
    name="norne-nb",
    pretty_name="the truncated version of the Bokmål part of NorNE",
    huggingface_id="ScandEval/norne-nb-mini",
    task=NER,
    languages=[NB],
    prompt_template="{text}",  # TODO
)


NORNE_NN_CONFIG = DatasetConfig(
    name="norne-nn",
    pretty_name="the truncated version of the Nynorsk part of NorNE",
    huggingface_id="ScandEval/norne-nn-mini",
    task=NER,
    languages=[NN],
    prompt_template="{text}",  # TODO
)


MIM_GOLD_NER_CONFIG = DatasetConfig(
    name="mim-gold-ner",
    pretty_name="the truncated version of MIM-GOLD-NER",
    huggingface_id="ScandEval/mim-gold-ner-mini",
    task=NER,
    languages=[IS],
    prompt_template="{text}",  # TODO
)


WIKIANN_FO_CONFIG = DatasetConfig(
    name="wikiann-fo",
    pretty_name="the truncated version of the Faroese part of WikiANN",
    huggingface_id="ScandEval/wikiann-fo-mini",
    task=NER,
    languages=[FO],
    prompt_template="{text}",  # TODO
)


SCALA_SV_CONFIG = DatasetConfig(
    name="scala-sv",
    pretty_name="The Swedish part of ScaLA",
    huggingface_id="ScandEval/scala-sv",
    task=LA,
    languages=[SV],
    prompt_template="{text}",  # TODO
)


SCALA_DA_CONFIG = DatasetConfig(
    name="scala-da",
    pretty_name="the Danish part of ScaLA",
    huggingface_id="ScandEval/scala-da",
    task=LA,
    languages=[DA],
    prompt_template="{text}",  # TODO
)


SCALA_NB_CONFIG = DatasetConfig(
    name="scala-nb",
    pretty_name="the Bokmål part of ScaLA",
    huggingface_id="ScandEval/scala-nb",
    task=LA,
    languages=[NB],
    prompt_template="{text}",  # TODO
)


SCALA_NN_CONFIG = DatasetConfig(
    name="scala-nn",
    pretty_name="the Nynorsk part of ScaLA",
    huggingface_id="ScandEval/scala-nn",
    task=LA,
    languages=[NN],
    prompt_template="{text}",  # TODO
)


SCALA_IS_CONFIG = DatasetConfig(
    name="scala-is",
    pretty_name="the Icelandic part of ScaLA",
    huggingface_id="ScandEval/scala-is",
    task=LA,
    languages=[IS],
    prompt_template="{text}",  # TODO
)


SCALA_FO_CONFIG = DatasetConfig(
    name="scala-fo",
    pretty_name="the Faroese part of ScaLA",
    huggingface_id="ScandEval/scala-fo",
    task=LA,
    languages=[FO],
    prompt_template="{text}",  # TODO
)


SCANDIQA_DA_CONFIG = DatasetConfig(
    name="scandiqa-da",
    pretty_name="the Danish part of the truncated version of ScandiQA",
    huggingface_id="ScandEval/scandiqa-da-mini",
    task=QA,
    languages=[DA],
    prompt_template="{text}",  # TODO
)


SCANDIQA_NO_CONFIG = DatasetConfig(
    name="scandiqa-no",
    pretty_name="the Norwegian part of truncated version of ScandiQA",
    huggingface_id="ScandEval/scandiqa-no-mini",
    task=QA,
    languages=[NB, NN],
    prompt_template="{text}",  # TODO
)


SCANDIQA_SV_CONFIG = DatasetConfig(
    name="scandiqa-sv",
    pretty_name="the Swedish part of the truncated version of ScandiQA",
    huggingface_id="ScandEval/scandiqa-sv-mini",
    task=QA,
    languages=[SV],
    prompt_template="{text}",  # TODO
)


NQII_CONFIG = DatasetConfig(
    name="nqii",
    pretty_name="Natural Questions in Icelandic",
    huggingface_id="ScandEval/nqii-mini",
    task=QA,
    languages=[IS],
    prompt_template="{text}",  # TODO
)


FOQA_CONFIG = DatasetConfig(
    name="???",
    pretty_name="???",
    huggingface_id="ScandEval/???",
    task=QA,
    languages=[FO],
    prompt_template="{text}",  # TODO
)


SPEED_CONFIG = DatasetConfig(
    name="speed",
    pretty_name="the speed estimation benchmark",
    huggingface_id="",
    task=SPEED,
    languages=list(get_all_languages().values()),
    prompt_template="{text}",  # TODO
)
