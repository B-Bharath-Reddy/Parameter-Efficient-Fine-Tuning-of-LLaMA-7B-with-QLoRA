# QLoRA Fine-Tuning Mistral 7B: Instruction Specialist

## Project Overview

This project focuses on fine-tuning the Mistral 7B Instruct model using QLoRA to create a lightweight, instruction-following language model that can run efficiently on limited GPU resources such as Google Colab. The goal was to enable the model to understand instructions and produce helpful, task-specific responses across a variety of domains including reasoning, summarization, explanation, translation, and simple problem-solving.

The fine-tuning process was carried out using the Alpaca dataset, a collection of instruction–input–answer samples. Each example contains an instruction, an optional input field, and an output that represents the correct response. By training on this format, the model learns to map instructions (and inputs when provided) to high-quality, coherent answers.

---

## Objective

The primary objective of this project was to build a smaller, low-cost variant of an instruction-tuned large language model using QLoRA. The goals included:

1. Adapting a large open-source model (Mistral 7B) to follow instructions more reliably.
2. Reducing training memory requirements by quantizing the base model to 4-bit.
3. Demonstrating an end-to-end supervised fine-tuning (SFT) workflow using PEFT, LoRA adapters, and TRL’s SFTTrainer.
4. Producing an inference-ready model capable of generating context-aware responses for custom tasks without retraining.

---

## What We Did

1. Installed the required libraries including Transformers, Accelerate, Bitsandbytes, Datasets, PEFT, and TRL.
2. Loaded the Alpaca-Cleaned dataset and reformatted each sample into a single text block containing the instruction, optional input, and answer.
3. Loaded the Mistral 7B Instruct base model in 4-bit quantized mode to fit within Google Colab GPU constraints.
4. Applied a LoRA configuration to enable parameter-efficient fine-tuning.
5. Fine-tuned the model using SFTTrainer with a small batch size and gradient accumulation to avoid memory issues.
6. Saved the trained LoRA adapters.
7. Loaded the adapters back onto the base model for inference and generated responses using a simple instruction prompt.

---

## Business Problem / Use Case

Large language models are powerful but expensive to fine-tune and deploy. Many organizations need customized models that understand specific instructions or domain requirements, but the full fine-tuning of 7B or larger models is not feasible due to hardware limitations.

This project demonstrates how:

1. A large model can be adapted using low-cost hardware.
2. Fine-tuning can be performed with minimal GPU memory.
3. A customized instruction-following model can be produced without retraining all parameters.

The result is a compact and efficient model suitable for tasks such as customer support automation, educational assistance, content generation, question answering, and lightweight internal AI tools.

---

## Challenges Faced

1. Limited GPU memory in free Google Colab required careful configuration such as 4-bit quantization, small batch sizes, and gradient accumulation.
2. Mistral 7B requires trust_remote_code for loading, which can lead to compatibility issues across Transformers versions.
3. Alpaca dataset contains mixed instruction types, requiring clean formatting to maintain consistent training quality.
4. Long training times on free-tier GPUs. A full epoch can take many hours due to T4 hardware limitations.
5. Ensuring that LoRA adapters correctly merge with the base model during inference.
6. Managing tokenization, padding, and correct instruction formatting for Mistral’s chat template.


