"""A module to provide dataclasses for fine-tuning workflow.

Authors:
    Komang Elang Surya Prawira (komang.e.s.prawira@gdplabs.id)
    Fachriza Dian Adhiatma (fachriza.d.adhiatma@gdplabs.id)

Reviewers:
    Kevin Yauris (kevin.yauris@gdplabs.id)
    Moch. Nauval Rizaldi Nasril (moch.n.r.nasril@gdplabs.id)
    Novan Parmonangan Simanjuntak (novan.p.simanjuntak@gdplabs.id)
    Pray Somaldo (pray.somaldo@gdplabs.id)
    Muhammad Afif Al Hawari (muhammad.a.a.hawari@gdplabs.id)

References:
    [1] https://huggingface.co/docs/transformers/perf_train_gpu_one
    [2] https://github.com/GDP-ADMIN/gen-ai-internal/blob/jupyter-notebooks/rlhf/truthful_qa/01_LoRa_SFT_Falcon_7B_Instruct.ipynb # pylint: disable=line-too-long # noqa: B950
    [3] https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/training/run_clm_sft_with_peft.py
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch
from peft import LoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from trl import PPOConfig


@dataclass
class DataArguments:
    """Dataclass for storing the arguments needed for handling the dataset."""

    dataset_path: str = field(
        metadata={
            "help": "Path to the CSV file used for fine-tuning. "
            "The CSV must contain 2 (two) columns, namely `prompt` and `expected_answer`"
        }
    )
    max_seq_length: Optional[int] = field(
        default=4096, metadata={"help": "The maximum sequence length of the inputs dataset"}
    )

    dataset_text_field: Optional[str] = field(
        default="prompt", metadata={"help": "The name of the column containing the input text."}
    )


@dataclass
class ModelArguments:
    """Dataclass for storing the arguments needed for model initialization."""

    # pylint: disable=too-many-instance-attributes
    # Disabling the pylint error is justified for this dataclass.
    # Eight attributes are reasonable and necessary in this case.
    pretrained_model: Union[PreTrainedModel, str] = field(
        metadata={
            "help": "The model to use for training. Can be pretrained model instance, The model ID of a pretrained"
            "model hosted inside a model repo on huggingface.co or a path to the directory containing model weights."
        }
    )
    tokenizer: Optional[Union[PreTrainedTokenizerBase, str]] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for the model. Can be a tokenizer instance or name/path to the"
            "directory containing model tokenizer. If not provided, the same tokenizer as `pretrained_model`"
            "will be loaded."
        },
    )
    torch_dtype: Optional[str or torch.dtype] = field(default="auto", metadata={"help": "PyTorch tensor dtype."})
    device_map: Optional[str or Dict[str, Union[int, str, torch.device]]] = field(
        default=None, metadata={"help": "A map that specifies where each submodule should go."}
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to allow for custom models defined on the Hub in their own modeling files."},
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "If True, will convert the loaded model into mixed-8bit quantized model."}
    )
    peft_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the PEFT model folder. " "If not None, model will be loaded using PEFT adapter"},
    )
    rm_peft_adapter_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the PEFT adapter and saved reward model folder."
            "If not None, model will be loaded using PEFT adapter"
        },
    )
    use_auth_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token to access private model."},
    )
    num_labels: Optional[int] = field(
        default=1,
        metadata={"help": "Number of neurons on the last dense layer. Only use this for reward model training."},
    )


@dataclass
class LoraArguments(LoraConfig):
    """Dataclass for storing the LoRA-specific arguments inheriting from LoraConfig."""

    task_type: Optional[str] = field(
        default="CAUSAL_LM", metadata={"help": "Task type to use.", "choices": ["CAUSAL_LM", "SEQ_2_SEQ_LM"]}
    )
    lora_rank: Optional[int] = field(default=8, metadata={"help": "Lora attention dimension."})
    target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "The names of the modules to apply Lora to. Should be differ based on the model architecture",
            "choices": ["query_key_value", "q, k, v", "q_proj, k_proj, v_proj"],
        },
    )
    lora_alpha: Optional[int] = field(default=32, metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "The dropout probability for Lora layers."})
    lora_bias: Optional[str] = field(
        default="none", metadata={"help": "Bias type for Lora.", "choices": ["none", "all", "lora_only"]}
    )


@dataclass
class LoraTrainingArguments(TrainingArguments):
    """Dataclass for storing the LoRA training-specific arguments, inheriting from TrainingArguments."""

    # pylint: disable=too-many-instance-attributes
    # Disabling the pylint error is justified for this dataclass.
    # Seventeen attributes are reasonable and necessary in this case.
    output_dir: Optional[str] = field(
        default="./output",
        metadata={help: "The output directory where the final model and checkpoints will be written."},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "The batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4,
        metadata={
            "help": "Number of updates steps to accumulate the gradients for, before performing a backward/update pass."
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."},
    )
    optim: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "The optimizer to use."})
    save_steps: Optional[int] = field(
        default=50, metadata={"help": "Number of updates steps before two checkpoint saves."}
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints and deletes the older checkpoints in output_dir"},
    )
    logging_steps: Optional[int] = field(default=50, metadata={"help": "Number of update steps between two logs."})
    learning_rate: Optional[float] = field(
        default=2e-4, metadata={"help": "The initial learning rate for AdamW optimizer."}
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."},
    )
    max_grad_norm: Optional[float] = field(
        default=0.3, metadata={"help": "Maximum gradient norm (for gradient clipping)."}
    )
    num_train_epochs: Optional[int] = field(default=4, metadata={"help": "Total number of training epochs to perform."})
    warmup_ratio: Optional[float] = field(
        default=0.03,
        metadata={"help": "Ratio of total training steps used for a linear warmup from 0 to learning_rate."},
    )
    group_by_length: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether or not to group together samples of roughly the same length in the training "
            "dataset (to minimize padding applied and be more efficient)."
        },
    )
    lr_scheduler_type: Optional[str] = field(
        default="constant",
        metadata={"help": "The scheduler type to use.", "choices": ["constant", "linear", "cosine", "polynomial"]},
    )
    resume_from_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the latest checkpoint folder. "
            "If not None, model will be continue training from checkpoint."
        },
    )
    remove_unused_columns: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to automatically remove the columns unused by the model forward method."},
    )


@dataclass
class PPOTrainingConfig(PPOConfig):
    """Dataclass for storing specific PPO training config, inheriting from PPOConfig."""

    # Common parameters
    model_name: Optional[str] = field(
        default=None, metadata={"help": "Name of model to use - used only for tracking purpose."}
    )
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Remove unused columns from the dataset if `datasets.Dataset` is used"}
    )
    # Hyperparameters
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate."})
    batch_size: int = field(default=1, metadata={"help": "Number of samples per optimisation step."})
    ppo_epochs: int = field(default=4, metadata={"help": "Number of optimisation epochs per batch of samples."})
