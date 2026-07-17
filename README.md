# QLoRA From Scratch: NF4 Quantization & LoRA Injection on Qwen2.5-7B-Instruct

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-blue)](https://huggingface.co/thefounder03/qlora-nl2sql-qwen2.5-7b)

This repository contains a PyTorch implementation of QLoRA (Quantized Low-Rank Adaptation) built from scratch—without relying on high-level wrappers like `peft` or `bitsandbytes`. The project demonstrates the mechanical implementation of 4-bit NormalFloat (NF4) quantization, 8-bit double quantization of scaling factors, and the injection of trainable low-rank adapters into a 7-billion parameter language model (`Qwen/Qwen2.5-7B-Instruct`), fine-tuned on the Spider text-to-SQL benchmark.

---

## 1. Project Summary

The core objective of this project was to manually implement the mathematical components of QLoRA and apply them to a real-world, instruction-tuned LLM. The pipeline consists of the following components:

*   **NF4 Codebook Construction:** Mathematical generation of the 16 quantile-based values for a standard normal distribution, structured asymmetrically to represent zero exactly.
*   **Block-Wise NF4 Quantization:** Splitting weights into blocks of 64 parameters, calculating the absolute maximum scale factor for each block, normalizing the weights, and mapping them to the nearest NF4 bin.
*   **Double Quantization (DQ):** Normalizing block-level scales (float32) by subtracting their mean, grouping them into blocks of 256, and quantizing them to 8-bit integers via uniform affine quantization to minimize memory overhead.
*   **Custom `QuantizedLinear` Layer:** A PyTorch module that stores weights in their compressed form (4-bit indices and 8-bit scale indices) and dequantizes them back to the active activation type (float16) on-the-fly during the forward pass.
*   **LoRA Injection:** A custom wrapper that intercepts the linear layers (`q_proj` and `v_proj` across all 28 attention blocks), introducing trainable low-rank matrices $A$ (initialized via Kaiming uniform) and $B$ (initialized to zero) with rank $r=8$ and scaling factor $\alpha=16$.
*   **Fine-Tuning:** End-to-end training of the adapter parameters on the Spider dataset for natural language to SQL translation.

---

## 2. Scope & Design Philosophy

This project was built as an educational codebase to develop a mechanical understanding of QLoRA's numerical boundaries. It is not intended to be a production-optimized library. 

### Scope Boundaries:
*   **Memory Management:** Paged optimizers are not built from scratch; we utilize `bitsandbytes`' `PagedAdamW` directly. Re-implementing paged optimizers requires CUDA-level unified memory allocation, which lies outside the scope of this PyTorch-level implementation.
*   **Performance:** Dequantization occurs on-the-fly in pure PyTorch/Python during the forward and backward passes. Without fused CUDA kernels (such as those in `bitsandbytes`), training throughput is lower. This was a deliberate tradeoff to preserve code clarity and readability.

---

## 3. Verification of the Quantization Math

Prior to model integration, the custom quantization modules were verified against naive linear quantization on synthetic tensors to evaluate numerical precision and memory compression.

### Test 1: NF4 vs. Naive INT4 Quantization
Tested on a large normal-distribution tensor ($1024 \times 1024$ dimensions, block size = 64):
*   **NF4 Mean Squared Error (MSE):** `0.0000033853`
*   **Naive INT4 Mean Squared Error (MSE):** `0.0000040374`
*   **Precision Improvement:** **16.15% reduction in MSE** when using quantile-based NormalFloat over uniform spacing.
*   **Weight Compression Ratio:** **7.11x** (4.19 MB float32 weight compressed to 0.59 MB of packed indices and scales).

### Test 2: Double Quantization Scale Compression
Tested on level-1 float32 scale factors (block size = 256):
*   **Scale bit-width reduction:** Compressed from 32-bit floats to 8-bit integers, plus block-level float32 scale metrics.
*   **Scale Memory Savings:** **74.6% reduction** in VRAM required for storing weight scales.

---

## 4. Fine-Tuning Progression

Fine-tuning a 7B parameter model under manual quantization involved three distinct runs to identify and resolve convergence and alignment failures.

### Run 1: Unstable Baseline (Schema-Blind)
*   **Configuration:** Schema-blind prompt (only `db_id` provided), `max_length=256`.
*   **Hyperparameters:** Learning rate = `2e-4`, no learning rate scheduler, no checkpoints saved during training.
*   **Results:**
    *   Training loss steadily decreased from `0.4663` to `0.3375`.
    *   Validation loss reached a minimum of `0.4503` at step 400, after which it diverged, rising to `0.54 - 0.61` by step 1000.
*   **Diagnosis:** The learning rate was too high for a 7B model under QLoRA, leading to overfitting. Because checkpointing was not implemented, the optimal model parameters at step 400 were lost.

### Run 2: Stable Baseline (Schema-Blind)
*   **Configuration:** Schema-blind prompt, `max_length=256`.
*   **Hyperparameters:** Lowered learning rate to `1e-4`, linear warmup (10% of steps) and decay scheduler, validation checkpointing.
*   **Results:**
    *   Training loss converged stably without divergence.
    *   Reached a best validation loss of **`0.4531` at step 850 / 1000**.
*   **Performance:** Evaluated to produce a baseline exact match of **10.00%** on the validation subset. Because the model was never shown the database schema, it wrote syntactically correct SQL but hallucinated column/table mappings.

### Run 3: Schema-Grounded Training (Active Checkpoint)
*   **Configuration:** Joined schema metadata from `richardr1126/spider-schema`, formatted using ChatML, and expanded context length to `max_length=1024` to prevent SQL target truncation.
*   **Hyperparameters:** Learning rate = `1e-4`, linear warmup/decay scheduler, validation checkpointing.
*   **Results:**
    *   The loss converged rapidly due to the schema context.
    *   Reached a best validation loss of **`0.2434` at step 350 / 1000**.
*   **Performance:** The **step 350 checkpoint** from this run was loaded for all final evaluation metrics in Section 6, achieving **44.00% quote-normalized exact match**.

---

## 5. The Schema-Grounding Failure & Fix

The 10.00% exact match baseline in Run 2 was caused by prompt schema blindness. The model wrote SQL without knowing what columns existed.

### The Grounding Fix
1.  **Metadata Join:** We integrated the `richardr1126/spider-schema` dataset, matching on `db_id` to retrieve table schemas, data types, and foreign key relations.
2.  **Context Expansion:** Schema details expanded prompt lengths. A token length audit of the training splits showed:
    *   *Average length:* 381.8 tokens.
    *   *95th percentile:* 814 tokens.
    *   *99th percentile:* 2,344 tokens.
    We set `max_length` to `1024` to avoid cutting off target queries. 164 of 7,000 training examples (oversized schemas like `baseball_1`) were filtered out of the dataset entirely, and 0 validation examples were dropped.
3.  **ChatML Formatting:** We formatted the prompts with Qwen's ChatML template using the tokenizer's chat template processor:

```python
def tokenize_example(example, tokenizer, max_length=1024):
    schema_text = format_schema(example["db_id"])
    prompt = build_prompt(example["question"], schema_text)
    
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answer_text = example["query"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    total_len = len(prompt_ids) + len(answer_ids)
    if total_len > max_length:
        return None

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    # Pad to max_length
    pad_len = max_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    labels    = labels    + [-100] * pad_len

    return {
        "input_ids": torch.tensor(input_ids),
        "labels":    torch.tensor(labels),
        "total_len": total_len,
    }
```

---

## 6. Evaluation Metrics

We evaluated the final schema-grounded Run 3 (Step 350) checkpoint on a representative subset of 300 validation examples across three distinct strictness thresholds:

| Metric | Normalization Rules | Score |
| :--- | :--- | :--- |
| **Raw Exact Match** | Character-for-character exact match (whitespace stripped) | **10.67%** (32 / 300) |
| **Case/Space Normalized** | Lowercased, multiple spaces collapsed to one | **35.33%** (106 / 300) |
| **Case/Space/Quote Normalized** | Casing and spacing collapsed; single and double quotes normalized | **44.00%** (132 / 300) |

*Important: These are three different scoring strictness levels calculated on the exact same set of predictions from the Run 3 (Step 350) checkpoint, not three different runs.*

### Limitation of Exact-Match Metrics
Exact match does not measure semantic equivalence. The model is penalized for structurally valid SQL queries that write a different but functionally equivalent logic.

#### Failure Example 1 (Join vs. Subquery):
*   **Gold :** `SELECT count(*) FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid WHERE T1.age  >  20`
*   **Pred :** `SELECT count(*) FROM has_pet WHERE Stuid IN (SELECT Stuid FROM student WHERE age  >  20)`
*   *Result:* Scored as **Incorrect** due to strict structural differences, despite returning identical records.

#### Failure Example 2 (Generation Truncation on Complex Filter):
*   **Gold :** `SELECT T1.fname ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'dog' AND T1.stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')`
*   **Pred :** `SELECT T1.fname ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T2.petid  =  T3.petid WHERE`
*   *Result:* Scored as **Incorrect** because the model generated query ran into generation token limits (`max_new_tokens=50`), cutting off prior to the target condition.

---

## 7. Error Analysis

An audit of the remaining 168 failed predictions (under Case/Space/Quote normalization) shows the following error distribution:

| Error Category | Occurrences |
| :--- | :--- |
| **Column/Table Name Mismatch** | 56 |
| **Redundant JOIN clause** | 32 |
| **Minor syntax / alias / spacing differences** | 28 |
| **Missing JOIN clause** | 22 |
| **Missing LIMIT clause** | 15 |
| **Redundant WHERE clause** | 12 |
| **Missing ORDER BY clause** | 11 |
| **Missing COUNT aggregation** | 9 |
| **Missing EXCEPT clause** | 7 |
| **Redundant MAX aggregation** | 6 |
| **Missing GROUP BY clause** | 5 |
| **Redundant GROUP BY clause** | 5 |
| **Redundant MIN aggregation** | 4 |
| **Missing WHERE clause** | 4 |
| **Redundant COUNT aggregation** | 3 |
| **Missing MIN aggregation** | 3 |
| **Missing INTERSECT clause** | 3 |
| **Redundant UNION clause** | 2 |
| **Redundant EXCEPT clause** | 2 |
| **Redundant INTERSECT clause** | 1 |
| **Missing SUM aggregation** | 1 |
| **Missing UNION clause** | 1 |
| **Redundant SUM aggregation** | 1 |

---

## 8. Known Limitations

*   **No Multi-Seed Averaging:** The performance metrics represent a single training run. Due to resource limits, we did not average over multiple random seeds to establish statistical variance.
*   **Evaluation Set Size:** Evaluation was executed on a subset of 300 dev examples, not the full Spider dev dataset.
*   **No Execution-Based Evaluation:** Accuracy is evaluated on text comparison, which does not account for execution equivalence (e.g. JOIN vs. subqueries). True execution accuracy is likely higher than the 44.00% text-normalized exact-match score.
*   **Pure PyTorch Dequantization:** Dequantizing full weights in PyTorch during the forward pass limits execution speed compared to fused C++/CUDA kernels (e.g. `bitsandbytes`).

---

## 9. Core Accomplishments

This project successfully implemented and verified:
1.  **NormalFloat (NF4) Quantization** from scratch with an MSE improvement of 16.15% over standard INT4.
2.  **8-bit Double Quantization** of weight scales, reducing scale memory footprints by 74.6%.
3.  **Low-Rank Adaptation (LoRA)** modules injected on top of the quantized weights of a 7B parameter base model.
4.  **Instability Resolution** through linear warmup/decay scheduling.
5.  **Schema Grounding Integration** leading to a 3.5x improvement in exact-match accuracy.
