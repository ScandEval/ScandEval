"""Build Icelandic and Faroese sentiment datasets by translating NoReC"""

from transformers import pipeline
from typing import Tuple
from pathlib import Path
import json
from tqdm.auto import tqdm


def build_norec_is_fo():
    """Translates the NoReC corpus into Icelandic and Norwegian"""

    # Initialise translation pipeline
    translation_model = "Helsinki-NLP/opus-mt-SCANDINAVIA-SCANDINAVIA"
    translator = pipeline("translation", translation_model)

    # Initialise translation helper function
    def translate(doc: str) -> Tuple[str, str]:
        tr_docs = [f">>is<< {doc}", f">>fo<< {doc}"]
        translations = [dct["translation_text"] for dct in translator(tr_docs)]
        return tuple(translations)

    # Load NoReC
    norec_dir = Path("datasets/norec")
    norec_train = (norec_dir / "train.jsonl").read_text().split("\n")
    norec_test = (norec_dir / "test.jsonl").read_text().split("\n")

    # Initialise NoReC-IS and NoReC-FO directories
    norec_is_dir = Path("datasets/norec_is")
    if not norec_is_dir.exists():
        norec_is_dir.mkdir()
    norec_fo_dir = Path("datasets/norec_fo")
    if not norec_fo_dir.exists():
        norec_fo_dir.mkdir()

    # Translate NoReC train
    for json_line in tqdm(norec_train):
        if json_line != "":
            record = json.loads(json_line)
            is_text, fo_text = translate(record["text"])

            with (norec_is_dir / "train.jsonl").open("a") as f:
                is_record = dict(record)
                is_record["text"] = is_text
                f.write(json.dumps(is_record) + "\n")

            with (norec_fo_dir / "train.jsonl").open("a") as f:
                fo_record = dict(record)
                fo_record["text"] = fo_text
                f.write(json.dumps(fo_record) + "\n")

    # Translate NoReC test
    for json_line in tqdm(norec_test):
        if json_line != "":
            record = json.loads(json_line)
            is_text, fo_text = translate(record["text"])

            with (norec_is_dir / "test.jsonl").open("a") as f:
                is_record = dict(record)
                is_record["text"] = is_text
                f.write(json.dumps(is_record) + "\n")

            with (norec_fo_dir / "test.jsonl").open("a") as f:
                fo_record = dict(record)
                fo_record["text"] = fo_text
                f.write(json.dumps(fo_record) + "\n")


if __name__ == "__main__":
    build_norec_is_fo()
