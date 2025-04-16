"""A module to provide an end-to-end fine-tuning workflow.

This module contains the main class to orchestrate the fine-tuning along with several supporting classes.

Authors:
    Komang Elang Surya Prawira (komang.e.s.prawira@gdplabs.id)
    Fachriza Dian Adhiatma (fachriza.d.adhiatma@gdplabs.id)

Reviewers:
    Kevin Yauris (kevin.yauris@gdplabs.id)
    Moch. Nauval Rizaldi Nasril (moch.n.r.nasril@gdplabs.id)
    Muhammad Afif Al Hawari (muhammad.a.a.hawari@gdplabs.id)
    Novan Parmonangan Simanjuntak (novan.p.simanjuntak@gdplabs.id)
    Pray Somaldo (pray.somaldo@gdplabs.id)

References:
    [1] https://huggingface.co/docs/transformers/main_classes/model
    [2] https://huggingface.co/docs/datasets/v1.11.0/loading_datasets.html
    [3] https://huggingface.co/docs/peft/package_reference/tuners
    [4] https://huggingface.co/docs/transformers/main_classes/trainer
    [5] https://github.com/GDP-ADMIN/gen-ai-internal/blob/jupyter-notebooks/rlhf/truthful_qa/01_LoRa_SFT_Falcon_7B_Instruct.ipynb # pylint: disable=line-too-long # noqa: B950
    [6] https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/training/run_clm_sft_with_peft.py
    [7] https://github.com/huggingface/transformers/issues/22794
"""

import logging
import multiprocessing
import random
from typing import Union

from datasets import load_dataset
from gdplabs_gen_ai_training.training_arguments import (
    DataArguments,
    LoraArguments,
    LoraTrainingArguments,
    ModelArguments,
)
from gdplabs_gen_ai_training.training_helpers import LoraConfigurator, ModelLoader, TrainerInitializer, setup_logging
from gdplabs_gen_ai_training.utils import check_gpu_available, check_ram_available, validate_args
from transformers import DataCollatorForLanguageModeling, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from trl import SFTTrainer as SFT_Trainer

logger = logging.getLogger(__name__)


class DatasetLoader:
    """DatasetLoader class for managing, loading, and formatting datasets with respect to a given tokenizer.

    The class is responsible for loading datasets from a specified CSV file path,
    shuffling the raw dataset, and formatting it according to the provided tokenizer.

    Attributes:
        data_args (DataArguments): The configuration object containing the necessary arguments for the dataset.
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer instance to format the dataset.
        dataset (Dataset): The loaded and formatted dataset.

    Methods:
        load_and_format_dataset(): Loads the dataset from the CSV file, shuffles it,
                                   and formats it using the given tokenizer.
    """

    def __init__(
        self, data_args: DataArguments, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    ) -> None:
        """Initializes a new instance of the DatasetLoader class.

        Args:
            data_args (DataArguments): The arguments needed for processing the data.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer to be used for tokenization.

        Attributes:
            dataset (Dataset): Attribute to store the dataset, initialized as None.
        """
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.dataset = None

    def load_and_format_dataset(self) -> None:
        """Loads and formats the dataset based on the provided data arguments and tokenizer.

        The method loads the dataset from the specified path in data_args, shuffles it,
        and then formats and tokenizes the prompts using the provided tokenizer and
        maximum sequence length from data_args. The formatted and tokenized dataset is
        then assigned to the dataset attribute of the object.

        Attributes:
            dataset (Dataset): Initialized with the formatted and tokenized dataset.

        Note:
            The raw dataset is loaded using all available CPU cores and is shuffled using a randomly
            generated seed between 0 and 2**32 - 1.
        """
        try:
            raw_dataset = load_dataset(
                "csv", data_files=self.data_args.dataset_path, num_proc=multiprocessing.cpu_count()
            )
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}") from e

        def concatenate_columns(data_point):
            data_point["prompt"] = f"""{data_point["prompt"]}{data_point["expected_answer"]}"""
            return data_point

        seed = random.randint(0, 2**32 - 1)
        self.dataset = raw_dataset["train"].shuffle(seed=seed).map(concatenate_columns)


class SFTComponents:
    """SFTComponents class responsible to initializes the components for SFTTrainer class.

    The class initializes model, dataset, LoRA configurator, and trainer initializer.

    The method logs information about the initialization process of each component and
    uses helper objects to load the model and tokenizer, load and format the dataset,
    configure LoRA, and initialize the trainer. After initializing all components successfully,
    a success log message is recorded.

    Attributes:
        model_loader (ModelLoader): Initialized with the model arguments and loads the model and tokenizer.
        dataset_loader (DatasetLoader): Initialized with the data arguments and tokenizer,
                                        then loads and formats the dataset.
        lora_configurator (LoraConfigurator): Initialized with the LoRA arguments and the model, then configures LoRA.
        trainer_initializer (TrainerInitializer): Initialized with the model, tokenizer, dataset,
                                                  and training arguments, then initializes the trainer.
    """

    def __init__(self, model_args, data_args, lora_args, training_args):
        """Initializes a new instance of the SFTComponents class.

        Args:
            model_args (ModelArguments): The model arguments.
            data_args (DataArguments): The data arguments.
            lora_args (LoraArguments): The LoRA configuration arguments.
            training_args (LoraTrainingArguments): The training settings arguments.

        Attributes:
            model_loader (ModelLoader): Attribute to store the model and tokenizer from ModelLoader.
            dataset_loader (DatasetLoader): Attribute to store the dataset from DatasetLoader.
            lora_configurator (LoraConfigurator): Attribute to store the model from LoraConfigurator.
            trainer_initializer (TrainerInitializer): Attribute to store the trainer from TrainerInitializer.
        """
        logger.info("Initializing components...")

        logger.info("Loading model...")
        self.model_loader = ModelLoader(model_args)
        self.model_loader.load_model_and_tokenizer()

        logger.info("Loading and formatting dataset...")
        self.dataset_loader = DatasetLoader(data_args, self.model_loader.tokenizer)
        self.dataset_loader.load_and_format_dataset()

        logger.info("Configuring Lora...")
        self.lora_configurator = LoraConfigurator(model_args, lora_args, self.model_loader.model)
        self.lora_configurator.configure_lora()

        logger.info("Initializing trainer...")
        self.trainer_initializer = TrainerInitializer(
            self.lora_configurator.model,
            self.model_loader.tokenizer,
            self.dataset_loader.dataset,
            data_args,
            training_args,
            SFT_Trainer,
            DataCollatorForLanguageModeling(self.model_loader.tokenizer, mlm=False),
        )
        self.trainer_initializer.initialize_trainer()


class SFTTrainer:
    """SFTTrainer class for initializing and training a model.

    This class is using LoRA (Low-Rank Adaptation) with the transformers library..

    Attributes:
        model_args (ModelArguments): Arguments related to the model.
        data_args (DataArguments): Arguments related to the training dataset.
        lora_args (LoraArguments): Arguments related to LoRA configuration.
        training_args (LoraTrainingArguments): Arguments related to training settings.

    Methods:
        train(): Trains the model using the initialized components and saves it for continue pretraining or inference.

    Note:
        As indicated in the TODO above, there's a planned investigation into the potential functional overlaps between
        this SFTTrainer class and the SFTTrainer class from the TRL library. Outcomes of this investigation may
        influence future updates or integrations. Contributors to this SDK will carry out this task.
    """

    def __init__(self, pretrained_model: Union[PreTrainedModel, str], dataset_path: str, **kwargs):
        """Initializes a new instance of the SFTTrainer class.

        During initialization, the method sets up logging, checks hardware requirements,
        validates and initializes arguments for model, data, LoRA, and LoRA training,
        and calls the SFTComponents class to initialize components. If GPU is not available,
        it raises an exception.

        Args:
            pretrained_model (Union[PreTrainedModel, str]): Pretrained model instance or the name/path of the model.
            dataset_path (str): The path to the dataset.
            **kwargs: Additional keyword arguments for initializing the arguments objects.

        Attributes:
            model_args (ModelArguments): Attribute to store the validated model-related arguments.
            data_args (DataArguments): Attribute to store the validated data-related arguments.
            lora_args (LoraArguments): Attribute to store the validated LoRA-related arguments.
            training_args (LoraTrainingArguments): Attribute to store the validated LoRA training-related arguments.
            resume_from_checkpoint_path (str): Attribute to store the resume from checkpoint path, initialized as None.
            components (SFTComponents): Attribute to store the initialized components.

        Note:
            The method logs info about the initialization steps and hardware requirements checks.
        """
        setup_logging()

        logger.info("Initializing and validating arguments...")
        self.model_args = validate_args(ModelArguments, dict({"pretrained_model": pretrained_model}, **kwargs))
        self.data_args = validate_args(DataArguments, dict({"dataset_path": dataset_path}, **kwargs))
        self.lora_args = validate_args(LoraArguments, kwargs)
        self.training_args = validate_args(LoraTrainingArguments, kwargs)

        self.resume_from_checkpoint_path = None
        self.components = SFTComponents(self.model_args, self.data_args, self.lora_args, self.training_args)

        logger.info("Checking hardware requirements...")
        gpu_info = check_gpu_available()
        logging.info(gpu_info)
        ram_info = check_ram_available()
        logging.info(ram_info)

        logger.info("All components initialized successfully!")

    def train(self) -> None:
        """Trains the initialized model using the configured trainer.

        The method performs gradient checkpointing if enabled, resumes from a checkpoint if specified,
        starts fine-tuning the model, and finally, saves the fine-tuned model to the specified output directory.

        Attributes:
            resume_from_checkpoint_path (str): Updated with the path provided in training arguments if resuming
                                               from a checkpoint is specified.

        Note:
            - If gradient checkpointing is enabled, the method attempts to make inputs require gradients.
            - The fine-tuned model is saved in the 'final' subdirectory of the specified output directory.
        """
        if self.training_args.gradient_checkpointing:
            if hasattr(self.components.trainer_initializer.model, "enable_input_require_grads"):
                self.components.trainer_initializer.model.enable_input_require_grads()
            elif hasattr(self.components.trainer_initializer.model, "get_input_embeddings"):

                def make_inputs_require_grad(_module, _input, _output):
                    _output.requires_grad_(True)

                self.components.trainer_initializer.model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        if self.training_args.resume_from_checkpoint_path is not None:
            logger.info("Loading checkpoint...")
            self.resume_from_checkpoint_path = self.training_args.resume_from_checkpoint_path

        logger.info("Starting fine-tuning...")
        self.components.trainer_initializer.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint_path)

        logger.info("Saving fine-tuned model...")
        self.components.trainer_initializer.model.save_pretrained(f"{self.training_args.output_dir}/final")
