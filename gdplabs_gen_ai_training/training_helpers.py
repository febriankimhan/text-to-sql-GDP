"""A module to provide shared training components.

This module contains classes for configuring model training.

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
    [8] https://huggingface.co/docs/trl/reward_trainer
    [9] https://github.com/GDP-ADMIN/gen-ai-internal/blob/jupyter-notebooks/rlhf/truthful_qa/02_LoRa_RM_Falcon_7B_Instruct.ipynb # pylint: disable=line-too-long # noqa: B950
"""

import logging
import os
import sys
from typing import Any, Union

import torch
from datasets import Dataset
from gdplabs_gen_ai_training.training_arguments import (
    DataArguments,
    LoraArguments,
    LoraTrainingArguments,
    ModelArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, RewardTrainer, SFTTrainer

module = __import__("transformers")


def setup_logging():
    """Configures the logging settings for the application.

    This function sets up basic logging configurations, ensuring that logs are
    displayed in a standardized format. It prints logs to standard output.
    """
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    log_level = logging.INFO
    handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(format=log_format, datefmt=date_format, level=log_level, handlers=[handler])


class ModelLoader:
    """ModelLoader class for managing and loading pretrained models and its tokenizers.

    Attributes:
        model_args (ModelArguments): A configuration object containing the required arguments for model and tokenizer.
        model (PreTrainedModel): The loaded model instance.
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The loaded tokenizer instance.
        is_reward_model (bool): Whether or not the model is loaded as a reward model.

    Methods:
        load_model_and_tokenizer(): Loads the model and tokenizer based on the provided model arguments.
    """

    def __init__(self, model_args: ModelArguments, is_reward_model: bool = False) -> None:
        """Initializes a new instance of the ModelLoader class.

        Args:
            model_args (ModelArguments): The arguments needed for the model.
            is_reward_model (bool): The argument for loading the final model as a reward model.

        Attributes:
            model_args (ModelArguments): Attribute to store the model arguments.
            model (PreTrainedModel): Attribute to store the model, initialized as None.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Attribute to store the tokenizer,
                                                                             initialized as None.
            is_reward_model (bool): Attribute to store the boolean value indicating \
                the loaded model is a reward model or not.
        """
        self.model_args = model_args
        self.model = None
        self.tokenizer = None
        self.is_reward_model = is_reward_model

    def load_model_and_tokenizer(self) -> None:
        """Loads the model and tokenizer based on the model arguments.

        The method initializes the model and tokenizer attributes of the object based on the
        pretrained model and tokenizer instance or name/path defined in model_args.

        Attributes:
            model (PreTrainedModel): Initialized with the model corresponding to the specified architecture.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Initialized with the tokenizer
                corresponding to the specified tokenizer instanze or name/path.

        Note:
            If the pad_token attribute of the tokenizer is None, it will be set to the eos_token of the tokenizer.
        """
        self.load_model()
        self.load_tokenizer()

    def load_model(self) -> None:
        """Loads the model based on the model arguments.

        The method initializes the model attribute of the object based on the
        pretrained model instance or name/path defined in model_args.

        Attributes:
            model (PreTrainedModel): Initialized with the model corresponding to the specified architecture.
        """
        if self.model_args.device_map is None:
            self.model_args.device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

        if isinstance(self.model_args.pretrained_model, PreTrainedModel):
            self.model = self.model_args.pretrained_model
        elif self.is_reward_model:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_args.pretrained_model,
                torch_dtype=self.model_args.torch_dtype,
                device_map=self.model_args.device_map,
                trust_remote_code=self.model_args.trust_remote_code,
                load_in_8bit=self.model_args.load_in_8bit,
                use_auth_token=self.model_args.use_auth_token,
                num_labels=self.model_args.num_labels,
            )
        else:
            config = AutoConfig.from_pretrained(
                self.model_args.pretrained_model, use_auth_token=self.model_args.use_auth_token
            )
            model_class = getattr(module, config.architectures[0])
            self.model = model_class.from_pretrained(
                self.model_args.pretrained_model,
                torch_dtype=self.model_args.torch_dtype,
                device_map=self.model_args.device_map,
                trust_remote_code=self.model_args.trust_remote_code,
                load_in_8bit=self.model_args.load_in_8bit,
                use_auth_token=self.model_args.use_auth_token,
            )

        if self.model_args.peft_adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model, self.model_args.peft_adapter_path, device_map=self.model_args.device_map
            )
            self.model = self.model.merge_and_unload()

        if self.model_args.rm_peft_adapter_path is not None and self.is_reward_model:
            self.model = PeftModel.from_pretrained(
                self.model, self.model_args.rm_peft_adapter_path, device_map=self.model_args.device_map
            )
            self.model.load_state_dict(torch.load(f"{self.model_args.rm_peft_adapter_path}/finetuned_rm"))

    def load_tokenizer(self) -> None:
        """Loads the tokenizer based on the tokenizer arguments.

        The method initializes the tokenizer attribute of the object based on the
        pretrained tokenizer name or path defined in model_args.

        Attributes:
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Initialized with the tokenizer
                corresponding to the specified tokenizer instance or name/path.

        Note:
            If the pad_token attribute of the tokenizer is None, it will be set to the eos_token of the tokenizer.
        """
        if isinstance(self.model_args.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            self.tokenizer = self.model_args.tokenizer
        else:
            if self.model_args.tokenizer is None:
                self.model_args.tokenizer = self.model.config._name_or_path
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_args.tokenizer,
                    use_auth_token=self.model_args.use_auth_token,
                    add_eos_token=True,
                )
            except Exception:
                # Disable TokenizerFast usage.
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_args.tokenizer_name_or_path,
                    use_auth_token=self.model_args.use_auth_token,
                    add_eos_token=True,
                    use_fast=False,
                )
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})


class LoraConfigurator:
    """LoraConfigurator class for managing and configuring the LoRA (Low-Rank Adaptation) parameters of a model.

    The class assists in setting up LoRA configurations for a model to aid in its adaptation and performance
    enhancement for specific tasks.

    Attributes:
        model_args (ModelArguments): A configuration object containing the required arguments for model and tokenizer.
        lora_args (LoraArguments): The configuration object containing the necessary arguments for LoRA configuration.
        model (PreTrainedModel): The model instance to which the LoRA configuration will be applied.
        return_lm_value_head (bool): Whether or not the model will be wrapped with AutoModelForCausalLMWithValueHead.

    Methods:
        configure_lora(): Configures the model with the provided LoRA settings and prepares it for k-bit training.
    """

    def __init__(
        self,
        model_args: ModelArguments,
        lora_args: LoraArguments,
        model: PreTrainedModel,
        return_lm_value_head: bool = False,
    ):
        """Initializes a new instance of the LoraConfigurator class.

        Args:
            model_args (ModelArguments): The arguments needed for the model.
            lora_args (LoraArguments): The arguments needed for configuring LoRA.
            model (PreTrainedModel): The pretrained model to which LoRA is applied.
            return_lm_value_head (bool): Wrap the model with AutoModelForCausalLMWithValueHead class or not.

        Attributes:
            lora_args (LoraArguments): Attribute to store the LoRA arguments.
            model (PreTrainedModel): Attribute to store the pretrained model that has been prepared for LoRA.
        """
        self.model_args = model_args
        self.lora_args = lora_args
        self.model = model
        self.return_lm_value_head = return_lm_value_head

    def configure_lora(self) -> None:
        """Configures LoRA (Low-Rank Adaptation) for the model based on the provided LoRA arguments.

        The method initializes a LoRA configuration with the parameters from lora_args and
        prepares the model for k-bit training. It then applies the LoRA configuration to the model.
        The model's caching mechanism is also disabled during this configuration.

        Attributes:
            model(PreTrainedModel): Initialized with the prepared and configured model for k-bit training with LoRA
                applied.

        Note:
            The target_modules for LoRA configuration are parsed from a comma-separated string from lora_args.

            The prepare_model_for_kbit_training() function is used only when the model is loaded in 8-bit mode.

            Set the return_lm_value_head to True if you want to train a PPO model.
        """
        target_modules = self.lora_args.target_modules.split(",") if self.lora_args.target_modules is not None else None

        lora_config = LoraConfig(
            r=self.lora_args.lora_rank,
            lora_alpha=self.lora_args.lora_alpha,
            lora_dropout=self.lora_args.lora_dropout,
            bias=self.lora_args.lora_bias,
            task_type=self.lora_args.task_type,
            modules_to_save=self.lora_args.modules_to_save,
            target_modules=target_modules,
        )
        if self.model_args.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False
        self.model = get_peft_model(self.model, lora_config)

        if self.return_lm_value_head:
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model)


class TrainerInitializer:
    """TrainerInitializer class for setting up and initializing the Trainer from the transformers library.

    The class consolidates and simplifies the process of trainer initialization by
    bundling the necessary components, i.e., the model, tokenizer, dataset, and training arguments.

    Attributes:
        model (PreTrainedModel): The model instance to be trained.
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer instance for tokenization.
        dataset (Dataset): The dataset on which the model will be trained.
        training_args (LoraTrainingArguments): The configuration object containing the required arguments for training.
        trainer (RewardTrainer): The initialized RewardTrainer instance from the trl library.
        data_collator (Any): The data collator used to train the model.
        ppo_config (PPOConfig): The configuration object containing the required arguments for PPO model training.

    Methods:
        initialize_trainer(): Sets up and initializes the Trainer using the provided components.
    """

    # pylint: disable=too-many-instance-attributes
    # Disabling the pylint error is justified for this dataclass.
    # Eight attributes are reasonable and necessary in this case.
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        dataset: Dataset,
        data_args: DataArguments,
        training_args: LoraTrainingArguments,
        trainer: Union[SFTTrainer, RewardTrainer, PPOTrainer],
        data_collator: Any,
        ppo_config: PPOConfig = None,
    ):
        """Initializes a new instance of the TrainerInitializer class.

        Args:
            model (PreTrainedModel): The pretrained model to be fine-tuned.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer for tokenization.
            dataset (Dataset): The dataset used for training.
            training_args (LoraTrainingArguments): The arguments for LoRA training.
            trainer (Union[SFTTrainer, RewardTrainer, PPOTrainer]): The trainer classes for performing model training.
            data_collator (Any): The data collator used to train the model.
            ppo_config (PPOConfig): The training configuration to train a PPO model.

        Attributes:
            model (PreTrainedModel): Attribute to store the provided pretrained model.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Attribute to store the provided tokenizer.
            dataset (Dataset): Attribute to store the provided dataset.
            training_args (LoraTrainingArguments): Attribute to store the provided training arguments.
            trainer (Union[Trainer, RewardTrainer]): Attribute to store the trainer instance, initialized as None.
            data_collator (Any): Attribute to store data collator used to train the model.
            ppo_config (PPOConfig): Attribute to store training configuration for PPO model training.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.training_args = training_args
        self.data_args = data_args
        self.trainer = trainer
        self.data_collator = data_collator
        self.ppo_config = ppo_config

    def initialize_trainer(self) -> None:
        """Initializes the trainer attribute with a specific Trainer instance based on its Type.

        The class is configured with the model, tokenizer, dataset, training arguments, and a specified
        data collator.

        The method initializes the trainer attribute with a Trainer instance configured
        with the provided model, tokenizer, dataset, training arguments, and the set up data collator.

        Attributes:
            trainer (Union[SFTTrainer, RewardTrainer, PPOTrainer]): Initialized with a Trainer instance.
        """
        if self.trainer == PPOTrainer:
            self.trainer = self.trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                dataset=self.dataset,
                data_collator=self.data_collator,
                config=self.ppo_config,
            )
        elif self.trainer == SFTTrainer:
            self.trainer = self.trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.dataset,
                args=self.training_args,
                data_collator=self.data_collator,
                dataset_text_field=self.data_args.dataset_text_field,
                max_seq_length=self.data_args.max_seq_length,
            )
        else:
            self.trainer = self.trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.dataset,
                args=self.training_args,
                data_collator=self.data_collator,
            )
