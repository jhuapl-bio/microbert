# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

from peft import TaskType
from peft import get_peft_model
from peft import LoraConfig
from peft import IA3Config


def prepare_peft_model(model_type, model, method="lora", lora_r=1):

    if method == "lora":
        if model_type == "NT":
            target_modules = ["query", "value"]
        elif model_type == "DNABERT":
            target_modules = ["Wqkv"]
        elif model_type == "HYENA":
            raise ValueError("HYENA model cannot be fine tuned using PEFT.")

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules,
            # modules_to_save=["intermediate"] # modules that are not frozen and updated during the training
        )

        ft_classifier = get_peft_model(
            model, peft_config
        )  # transform our classifier into a peft model
        ft_classifier.print_trainable_parameters()

    if method == "ia3":
        if model_type == "NT":
            target_modules = ["query", "value", "attention.output.dense"]
            feedforward_modules = ["attention.output.dense"]
        elif model_type == "DNABERT":
            target_modules = ["Wqkv", "attention.output.dense"]
            feedforward_modules = ["attention.output.dense"]
        elif model_type == "HYENA":
            raise ValueError("HYENA model cannot be fine tuned using PEFT.")

        peft_config = IA3Config(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            target_modules=target_modules,
            feedforward_modules=feedforward_modules,
            # modules_to_save=["intermediate"] # modules that are not frozen and updated during the training
        )
        ft_classifier = get_peft_model(
            model, peft_config
        )  # transform our classifier into a peft model
        ft_classifier.print_trainable_parameters()

    return ft_classifier
