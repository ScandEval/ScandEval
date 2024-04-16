"""Gradio app for conducting human evaluation of the tasks."""
# mypy: disable-error-code="name-defined"

import logging
from functools import partial

import click
import gradio as gr
from scandeval.benchmark_config_factory import build_benchmark_config
from scandeval.config import ModelConfig
from scandeval.dataset_configs import SPEED_CONFIG, get_all_dataset_configs
from scandeval.dataset_factory import DatasetFactory
from scandeval.enums import Framework, ModelType
from scandeval.utils import create_model_cache_dir, enforce_reproducibility
from transformers import AutoConfig, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TITLE = "ScandEval Human Evaluation"
DESCRIPTION = """
In this app we will evaluate your performance on a variety of tasks, with the goal of
comparing human performance to language model performance.

When you select a language and a task then you will be given a brief description of the
task, as well as examples of how to solve it. Please read through these examples before
proceeding with the task.

Please do not use any additional aids (such as search engines) when completing these
tasks.
"""


@click.command()
@click.option(
    "--annotator_id",
    "-id",
    type=int,
    required=True,
    help="""The annotator ID to use for the evaluation. Needs to be between 0 and 10,
    inclusive.""",
)
def main(annotator_id) -> None:
    """Start the Gradio app for human evaluation."""
    dataset_configs = get_all_dataset_configs()
    tasks = sorted(
        {
            cfg.task.name.replace("-", " ").title()
            for cfg in dataset_configs.values()
            if cfg != SPEED_CONFIG
        }
    )
    languages = sorted(
        {
            language.name
            for cfg in dataset_configs.values()
            if cfg != SPEED_CONFIG
            for language in cfg.languages
            if language.name not in {"Norwegian Bokmål", "Norwegian Nynorsk"}
        }
    )

    with gr.Blocks(title=TITLE, theme=gr.themes.Monochrome()) as demo:
        gr.components.HTML(f"<center><h1>{TITLE}</h1></center>")
        gr.components.Markdown(DESCRIPTION)
        with gr.Row(variant="panel"):
            language_dropdown = gr.Dropdown(label="Language", choices=languages)
            task_dropdown = gr.Dropdown(label="Task", choices=tasks)
            dataset_dropdown = gr.Dropdown(label="Dataset", choices=[""])
        with gr.Row(variant="panel"):
            with gr.Column():
                task_examples = gr.Markdown("Task Examples")
            with gr.Column():
                question = gr.Markdown("Question")
                answer = gr.Textbox(label="Answer", interactive=True)
                submit_button = gr.Button("Submit")

        language_dropdown.change(
            fn=update_dataset_choices,
            inputs=[language_dropdown, task_dropdown],
            outputs=dataset_dropdown,
        )
        task_dropdown.change(
            fn=update_dataset_choices,
            inputs=[language_dropdown, task_dropdown],
            outputs=dataset_dropdown,
        )
        dataset_dropdown.change(
            fn=partial(update_dataset, iteration=annotator_id),
            inputs=dataset_dropdown,
            outputs=[task_examples, question, answer],
        )
        submit_button.click(
            fn=partial(submit_answer, annotator_id=annotator_id),
            inputs=[dataset_dropdown, task_examples, question, answer],
            outputs=[question, answer],
        )
        answer.submit(
            fn=partial(submit_answer, annotator_id=annotator_id),
            inputs=[dataset_dropdown, task_examples, question, answer],
            outputs=[question, answer],
        )

    demo.launch()


def update_dataset_choices(language: str, task: str) -> gr.Dropdown:
    """Update the dataset choices based on the selected language and task.

    Args:
        language:
            The language selected by the user.
        task:
            The task selected by the user.

    Returns:
        A list of dataset names that match the selected language and task.
    """
    if language is None or task is None:
        return gr.Dropdown(label="Dataset", choices=[])

    dataset_configs = [
        cfg
        for cfg in get_all_dataset_configs().values()
        if language in {language.name for language in cfg.languages}
        and task.lower().replace(" ", "-") == cfg.task.name
    ]
    assert len(dataset_configs) > 0

    choices = sorted([cfg.name for cfg in dataset_configs])

    logger.info(
        f"User selected {language} and {task}, which resulted in the datasets "
        f"{choices}, with {choices[0]} being chosen by default."
    )

    return gr.Dropdown(label="Dataset", choices=choices, value=choices[0])


def update_dataset(dataset_name: str, iteration: int) -> tuple[str, str, str]:
    """Update the dataset based on a selected dataset name.

    Args:
        dataset_name:
            The dataset name selected by the user.
        iteration:
            The iteration index of the datasets to evaluate.

    Returns:
        A tuple (task_examples, question, answer) for the selected
        dataset.
    """
    if not dataset_name:
        return "", "", ""

    logger.info(f"User selected dataset {dataset_name} - loading dataset...")

    benchmark_config = build_benchmark_config(
        progress_bar=False,
        save_results=True,
        task=None,
        dataset=None,
        language=[
            language.code
            for cfg in get_all_dataset_configs().values()
            for language in cfg.languages
        ],
        model_language=None,
        dataset_language=None,
        framework=None,
        device=None,
        batch_size=1,
        evaluate_train=False,
        raise_errors=False,
        cache_dir=".scandeval_cache",
        token=None,
        openai_api_key=None,
        prefer_azure=False,
        azure_openai_api_key=None,
        azure_openai_endpoint=None,
        azure_openai_api_version=None,
        force=False,
        verbose=False,
        trust_remote_code=False,
        load_in_4bit=None,
        use_flash_attention=None,
        clear_model_cache=False,
        only_validation_split=True,
        few_shot=True,
        num_iterations=10,
        run_with_cli=True,
    )
    dataset_factory = DatasetFactory(benchmark_config=benchmark_config)
    dataset_config = get_all_dataset_configs()[dataset_name]

    model_id = f"human-{iteration}"
    model_config = ModelConfig(
        model_id=model_id,
        revision="main",
        framework=Framework.API,
        task="text-generation",
        languages=dataset_config.languages,
        model_type=ModelType.LOCAL,
        model_cache_dir=create_model_cache_dir(
            cache_dir=benchmark_config.cache_dir, model_id=model_id
        ),
    )

    dummy_hf_model_id = "mistralai/Mistral-7B-v0.1"
    dummy_hf_config = AutoConfig.from_pretrained(dummy_hf_model_id)
    dummy_tokenizer = AutoTokenizer.from_pretrained(dummy_hf_model_id)

    global active_dataset
    global sample_idx
    sample_idx = 0
    benchmark_dataset = dataset_factory.build_dataset(dataset=dataset_config)
    rng = enforce_reproducibility(framework=Framework.PYTORCH)
    train, val, tests = benchmark_dataset._load_data(rng=rng)
    _, _, tests = benchmark_dataset._load_prepared_data(
        train=train,
        val=val,
        tests=tests,
        model_config=model_config,
        hf_model_config=dummy_hf_config,
        tokenizer=dummy_tokenizer,
        benchmarking_generative_model=True,
    )

    active_dataset = tests[iteration].add_column(
        name="answer", column=[None] * len(tests[iteration])
    )

    task_examples, question = example_to_markdown(example=active_dataset[0])
    answer = ""

    logger.info(
        f"Loaded dataset {dataset_name}, with the following task examples:\n\n"
        f"{task_examples}"
    )

    return task_examples, question, answer


def submit_answer(
    dataset_name: str, task_examples: str, question: str, answer: str, annotator_id: int
) -> tuple[str, str]:
    """Submit an answer to the dataset.

    Args:
        dataset_name:
            The name of the dataset.
        task_examples:
            The task examples for the dataset.
        question:
            The question for the dataset.
        answer:
            The answer to the question.
        annotator_id:
            The annotator ID for the evaluation.

    Returns:
        A tuple (question, answer), with `question` being the next question, and
        `answer` being an empty string.
    """
    gr.Info("Submitted!")

    global active_dataset
    global sample_idx

    # Store the user's answer
    answers = active_dataset["answer"]
    answers[sample_idx] = answer
    active_dataset = active_dataset.remove_columns("answer").add_column(
        name="answer", column=answers
    )
    logger.info(f"User submitted the answer {answer!r} to the question {question!r}.")

    try:
        # Attempt to get the next question
        sample_idx += 1
        _, question = example_to_markdown(example=active_dataset[sample_idx])

    except IndexError:
        gr.Info("No more questions left in this dataset. Please select a new dataset.")
        question = ""

    return question, ""


def example_to_markdown(example: dict) -> tuple[str, str]:
    """Convert an example to a Markdown string.

    Args:
        example:
            The example to convert.

    Returns:
        A tuple (task_examples, question) for the example.
    """
    task_examples: str | list[str] = [
        sample.replace("\n", "\n\n") for sample in example["text"].split("\n\n")[:-1]
    ]
    task_examples = "\n\n**Example**\n\n".join(task_examples)

    question = "**Question**\n\n"
    question += "\n\n".join(example["text"].split("\n\n")[-1].split("\n")[:-1])
    question += "\n\n" + example["text"].split("\n\n")[-1].split("\n")[-1]

    return task_examples, question


if __name__ == "__main__":
    main()
