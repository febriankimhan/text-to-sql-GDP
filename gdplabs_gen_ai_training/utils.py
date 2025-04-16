"""A module to provide several helper functions for fine-tuning workflow.

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
    [1] https://huggingface.co/docs/transformers/v4.18.0/en/performance
    [2] https://github.com/GDP-ADMIN/gen-ai-internal/blob/jupyter-notebooks/rlhf/truthful_qa/01_LoRa_SFT_Falcon_7B_Instruct.ipynb # pylint: disable=line-too-long # noqa: B950
    [3] https://github.com/GDP-ADMIN/gen-ai-internal/blob/jupyter-notebooks/rlhf/truthful_qa/02_LoRa_RM_Falcon_7B_Instruct.ipynb # pylint: disable=line-too-long # noqa: B950
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Union

import torch
from psutil import virtual_memory
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


def format_prompt(data_point: Dict[str, str]) -> str:
    """Concatenates the 'prompt' and 'expected_answer' fields of a data point and returns the formatted string.

    Args:
        data_point (Dict[str, str]): A dictionary containing at least the fields 'prompt' and 'expected_answer'.
                                     Additional fields, if present, are ignored.

    Returns:
        str: The formatted string obtained by concatenating the 'prompt' and 'expected_answer' fields of the data point.

    Example:

    .. code-block:: python

        >>> data_point = {'prompt': 'What is the capital of France? ', 'expected_answer': 'Paris'}
        >>> formatted_prompt = format_prompt(data_point)
        >>> print(formatted_prompt)
        What is the capital of France? Paris
    """
    return f"""{data_point["prompt"]}{data_point["expected_answer"]}"""


def format_and_tokenize_prompt(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], max_seq_length: int, data_point: Dict[str, str]
) -> Dict[str, List[int]]:
    """Formats a data point into a prompt, tokenizes the prompt, and returns the tokenized result.

    Given a data point represented as a dictionary containing 'prompt' and 'expected_answer', this function formats
    the data point into a string prompt and then tokenizes it. The resulting tokenized prompt is truncated or padded
    as needed to ensure its length does not exceed the specified maximum sequence length.

    Args:
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer used to tokenize the prompt.
        max_seq_length (int): The maximum allowed sequence length for the tokenized prompt.
        data_point (Dict[str, str]): A dictionary containing the data point,
                                     typically includes 'prompt' and 'expected_answer'.

    Returns:
        Dict[str, List[int]]: The tokenized prompt represented as a dictionary
                              including 'input_ids' and 'attention_mask'.

    Example:

    .. code-block:: python

        >>> data_point = {'prompt': 'What is the capital of France?', 'expected_answer': 'Paris'}
        >>> tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')
        >>> tokenized_prompt = format_and_tokenize_prompt(tokenizer, 50, data_point)
        >>> print(tokenized_prompt)
        {'input_ids': [101, 2054, 2003, 1996, 3007, 1997, 2605, 1029, 102, 3000, 102],
        'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    """
    prompt = format_prompt(data_point)
    tokenized_prompt = tokenizer(prompt, truncation=True, max_length=max_seq_length)
    return tokenized_prompt


def create_comparison_data(dataset: Dict[str, List[str]], split: str = "train") -> List[Dict[str, str]]:
    """Format the given dataset into string of prompt-chosen response or prompt-rejected response pair.

    Args:
        dataset (Dict[str, List[str]]): The dataset that will be formatted into string of prompt-response pair
        split (string): Type of split in Hugging Face Dataset

    Returns:
        List[Dict[str, str]]: List of pairs chosen and rejected prompt-response.

    Example:

    .. code-block:: python

        >>> dataset = {
            "prompt": ["What happens to you if you eat watermelon seeds?", ...]
            "chosen": ["The watermelon seeds pass through your digestive system", ...]
            "rejected": ["Nothing happens", ...]
        }
        >>> formatted_dataset = create_comparison_data(dataset)
        >>> print(formatted_dataset)
        [{"chosen": "What happens to you if you eat watermelon seeds?
        The watermelon seeds pass through your digestive system"},
        {"rejected": "What happens to you if you eat watermelon seeds? Nothing happens"}, ...]
    """
    pairs = []

    for sample in tqdm(dataset[split]):
        pair = {}
        prompt = sample["prompt"]
        chosen_response = sample["chosen"]
        rejected_response = sample["rejected"]
        pair["chosen"] = prompt + " " + chosen_response
        pair["rejected"] = prompt + " " + rejected_response
        pairs.append(pair)

    return pairs


def format_and_tokenize_prompt_rm(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], max_seq_length: int, data_point: Dict[str, str]
) -> Dict[str, List[int]]:
    """Tokenizes the prompt-response pair, and returns the tokenized result.

    Given a data point represented as a dictionary containing 'chosen' and 'rejected' prompt-response pair,
    this function tokenizes them. The resulting tokenized prompt is truncated or padded as needed
    to ensure its length does not exceed the specified maximum sequence length.

    Args:
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer used to tokenize the prompt.
        max_seq_length (int): The maximum allowed sequence length for the tokenized prompt.
        data_point (Dict[str, str]): A dictionary containing the data point,
                                     typically includes 'chosen' and 'rejected'.

    Returns:
        Dict[str, List[int]]: The tokenized prompt represented as a dictionary
                              including 'input_ids_chosen', 'attention_mask_chosen',
                              'input_ids_rejected' and 'attention_mask_rejected'.
    """
    chosen_tokens = tokenizer(data_point["chosen"], truncation=True, max_length=max_seq_length)
    rejected_tokens = tokenizer(data_point["rejected"], truncation=True, max_length=max_seq_length)

    return {
        "input_ids_chosen": chosen_tokens["input_ids"],
        "attention_mask_chosen": chosen_tokens["attention_mask"],
        "input_ids_rejected": rejected_tokens["input_ids"],
        "attention_mask_rejected": rejected_tokens["attention_mask"],
    }


def data_collator_ppo(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Custom data collator for PPO training.

    Notes:
    Based on our experiments, the custom data collator is needed for training PPO model.
    The default data collator will lead to this error during training \
    "ValueError: queries must be a list of tensor - got <class 'torch.Tensor'>"
    """
    collated = {key: [d[key] for d in data] for key in data[0]}
    return collated


def check_gpu_available() -> None:
    """Validates the availability of a GPU and returns its name if it is accessible.

    This function checks if a GPU is available using PyTorch and returns its name if available.
    If not, it raises a RuntimeError indicating that no GPU is connected.

    Example:

    .. code-block:: python

        >>> check_gpu_available()
        Connected to GPU: Tesla K80

    Raises:
        RuntimeError: If no GPU is detected or connected.

    Note:
        Utilizing this function at the beginning of scripts or notebooks ensures that the necessary hardware is
        available.
    """
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        return f"Connected to GPU: {device}"
    else:
        raise RuntimeError("Not connected to a GPU!")


def check_ram_available() -> None:
    """Checks the available RAM, returns its size in gigabytes, and validates if it meets the required threshold.

    This function retrieves the total available RAM, returns it, and raises a RuntimeError if the available RAM is below
    20 gigabytes, otherwise, it confirms that a high-RAM runtime is being used.

    Example:

    .. code-block:: python

        >>> check_ram_available()
        Your runtime has 30.0 gigabytes of available RAM

    Raises:
    RuntimeError: If the available RAM is less than 20 gigabytes.
    """
    ram_gb = virtual_memory().total / 1e9

    if ram_gb < 20:
        raise RuntimeError("Not using a high-RAM runtime")
    else:
        return f"Your runtime has {ram_gb:.1f} gigabytes of available RAM"


def print_gpu_utilization() -> None:
    """Returns the current memory utilization of the GPU in megabytes.

    This function initializes NVML, retrieves the handle for the first GPU device, queries
    its memory information, and then returns the occupied GPU memory in MB.

    Example:
        Below is an example to show GPU utilization

    .. code-block:: python

        >>> print_gpu_utilization()
        GPU memory occupied: 2000 MB.

    Note:
        This function is useful for monitoring GPU memory utilization during model training
        to manage resources efficiently. It provides a quick snapshot of the amount
        of GPU memory currently being used.
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"GPU memory occupied: {info.used // 1024 ** 2} MB."


def print_trainable_parameters(model: PreTrainedModel) -> None:
    """Prints the number and percentage of trainable parameters in the provided model.

    This function calculates the number of trainable and total parameters and then prints
    this information along with the percentage of trainable parameters.

    Args:
    model (PreTrainedModel): The model for which to print trainable parameters information.

    Example:
    Below is an example to show the number and percentage of trainable parameters

    .. code-block:: python

        >>> model = SomePreTrainedModel()
        >>> print_trainable_parameters(model)
        Trainable params: 1000
        All params: 2000
        Trainable: 50.0 %

    Note:
        This function is useful for understanding the proportion of parameters in the model
        that can be trained.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params}",
        f"All params: {all_param}",
        f"Trainable: {round(100 * trainable_params / all_param, 3)} %",
        sep="\n",
    )


def validate_args(cls: type(dataclass), kwargs) -> type(dataclass):
    """Validates and initializes an instance of a dataclass based on the provided keyword arguments.

    Args:
        cls (type(dataclass)): The dataclass type to be initialized.
        kwargs (Dict): The dictionary of keyword arguments to be validated and used for initialization.

    Returns:
        An instance of the specified dataclass, initialized with the validated keyword arguments.

    Note:
        - Only keyword arguments that match the fields of the dataclass are used for initialization.
        - Extra keyword arguments are filtered out and do not affect the initialization.
    """
    return cls(**{k: v for k, v in kwargs.items() if k in {f.name for f in fields(cls)}})
