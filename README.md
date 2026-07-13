# qlora-nl2sql: NF4 Quantization from Scratch

This repository contains a manual, first-principles implementation of NormalFloat4 (NF4) quantization and dequantization built using PyTorch. This is a core component of a larger project designed to fine-tune a large language model (e.g., Qwen2.5-3B or similar) on the Spider text-to-SQL dataset using QLoRA. This project is a scaled-up successor to `nl2sql-lora`, which applied standard LoRA to a smaller Qwen2.5-0.5B model.

To establish a deep, mechanical understanding of quantization before relying on pre-built libraries, this implementation was built entirely from scratch with no dependency on `bitsandbytes`.

---

## 1. What is NF4 and Why it Beats Naive Uniform INT4

Quantization maps continuous high-precision values (like 32-bit floats) to a discrete set of lower-bit representations (like 4-bit indices). 

### The Gaussian Weight Distribution
In deep neural networks, weights are not uniformly distributed. Instead, weight parameters closely follow a zero-centered normal (Gaussian) distribution, where the vast majority of weights cluster tightly around zero, with very few weights at the outer extremes.

```
       Uniform INT4 Spacing:  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
       
                         * *
                       *     *
                     *         *
                   *             *
                  *               *
       NF4 Spacing: || |  |   |    |      |      |       |       |      |      |    |   |  | ||
```

### Quantile Quantization (NF4)
* **Naive Uniform INT4** spaces its 16 quantization bins evenly across the range of the tensor. This is highly inefficient because it allocates the same representation capacity to the sparse tails of the distribution as it does to the dense center, leading to high quantization error where most of the weights reside.
* **NormalFloat4 (NF4)** optimizes for Gaussian data by placing its 16 quantization levels at the **quantiles** of a standard normal distribution $N(0, 1)$. This concentrates the precision (closer spacing between bins) around zero, where the bulk of the weights live, and spaces them wider apart at the extremes.

### Asymmetric Codebook Structure
The NF4 codebook uses a specific asymmetric layout consisting of exactly 16 values:
* **8 negative values**
* **1 exact zero**
* **7 positive values**

Having an **exact zero** in the codebook is critical. Zero-valued weights or activations are extremely common (e.g., due to ReLU, padding, or general sparsity), and any quantization error around zero can significantly alter the model's behavior and performance.

---

## 2. Block-Wise Scaling

Because the standard normal quantiles are static, the NF4 codebook is defined once and normalized to the range $[-1, 1]$. To quantize arbitrary weight tensors, the tensors must be normalized to fit this range:

1. **The Outlier Problem:** If we normalize a whole tensor using a single global scale factor (the absolute maximum value of the entire tensor), a single extreme outlier will compress all other normal weights into a tiny range near zero, ruining precision for the entire tensor.
2. **Block-Wise Quantization:** To mitigate this, the tensor is flattened and partitioned into independent blocks of a fixed size (default `block_size=64`). 
3. **Per-Block Scales:** For each block, we compute a local scale factor:
   $$\text{scale} = \max(|x_i|)$$
   Each block is normalized individually by its scale before finding the nearest codebook representation. The scale factor is saved as a 32-bit float alongside the 4-bit quantized indices.

---

## 3. Implementation Summary

The quantization and dequantization functions are implemented with vectorized PyTorch operations, avoiding slow Python loops:

* **Codebook Construction:** Constructed using `scipy.stats.norm.ppf` (the percent point function / inverse cumulative distribution function) to find the precise theoretical normal quantiles, followed by normalizations.
* **Block Reshaping:** Input tensors are flattened, padded to be divisible by the block size, and reshaped into blocks using `.view(-1, block_size)`.
* **Nearest-Match Mapping:** Normal weights are matched to the closest NF4 codebook value using broadcasted distance computation (`(normalized.unsqueeze(-1) - codebook.view(1, 1, -1)).abs()`) and finding the minimum index via `.argmin(dim=-1)`.

### Function Signatures

Below are the signatures for the primary functions:

```python
import torch

def make_nf4_codebook() -> torch.Tensor:
    """
    Builds the 16-value NF4 codebook: quantiles of a standard normal
    distribution, asymmetric (8 negative, exact 0, 7 positive) so that
    zero is exactly representable.
    
    Returns:
        torch.Tensor: Sorted NF4 codebook of shape (16,) normalized to [-1, 1].
    """

def quantize_nf4(
    tensor: torch.Tensor, 
    codebook: torch.Tensor, 
    block_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Size, int]:
    """
    Quantizes a float tensor to 4-bit NF4 indices, block-wise.
    
    Args:
        tensor: The input float32 tensor to quantize.
        codebook: The 16-value NF4 codebook tensor.
        block_size: Number of elements per independent quantization block.
        
    Returns:
        indices: uint8 tensor of shape (num_blocks, block_size) containing values 0-15.
        scales: float32 tensor of shape (num_blocks,) containing the absmax scale for each block.
        orig_shape: The original shape of the input tensor (needed for dequantization).
        orig_numel: The original number of elements in the input tensor (needed for padding removal).
    """

def dequantize_nf4(
    indices: torch.Tensor, 
    scales: torch.Tensor, 
    codebook: torch.Tensor, 
    orig_shape: torch.Size, 
    orig_numel: int, 
    block_size: int = 64
) -> torch.Tensor:
    """
    Reconstructs the original float32 tensor from NF4 indices and block scales.
    
    Args:
        indices: uint8 tensor containing NF4 codebook indices.
        scales: float32 tensor containing per-block scale factors.
        codebook: The 16-value NF4 codebook tensor.
        orig_shape: The target shape of the reconstructed tensor.
        orig_numel: The original element count before padding.
        block_size: Number of elements per quantization block.
        
    Returns:
        torch.Tensor: Reconstructed float32 tensor matching the original shape.
    """
```

---

## 4. Empirical Validation

We verified the implementation using two tests:

### Small-Tensor Sanity Check
* **Setup:** A $4 \times 8$ tensor (32 values) with a block size of 8.
* **Result:** Reconstruction works correctly and maps block scales and indices properly. Due to the small sample size, statistical advantages of NF4 are not yet representative.

### Large-Tensor Benchmark
* **Setup:** A $1024 \times 1024$ tensor ($1,048,576$ values) sampled from a Gaussian distribution $N(0, 0.02)$ to simulate a realistic weight matrix.
* **Block Size:** 64
* **Metrics:**
  * **NF4 Mean Squared Error (MSE):** `0.0000033853`
  * **Naive Uniform INT4 MSE:** `0.0000040374`
  * **Reconstruction Accuracy:** **NF4 reduces MSE by 16.15%** compared to naive uniform INT4.
* **Compression Performance:**
  * **Original (fp32) size:** `4.19 MB`
  * **Quantized packed size:** `0.59 MB` (calculated as 4-bit per-value indices + 32-bit float scale per block)
  * **Compression Ratio:** **7.11x** compression

---

## 5. First-Principles Implementation

This implementation is a deliberate educational exercise. Instead of treating double quantization and block-wise scaling as black-box configurations inside `bitsandbytes`, implementing NF4 from scratch using standard PyTorch operations exposes the exact mathematical and storage mechanics underlying QLoRA. This step ensures complete transparency of the quantization pipeline before scaling up to LLM fine-tuning.

---

## 6. Next Steps

To complete the full custom QLoRA pipeline, the next milestones are:

1. **Double Quantization:** Compress the per-block FP32 scale factors themselves (e.g., quantizing 32-bit scale blocks to 8-bit floats with a second block size of 256) to yield further memory savings.
2. **Paged Optimizers:** Integrate page-locked memory managers to handle spike memory states during backward passes and avoid Out-Of-Memory (OOM) errors.
3. **Full QLoRA Fine-tuning:** Hook the custom quantized linear layers into a parameter-efficient fine-tuning loop targeting a Qwen2.5-3B model on the Spider text-to-SQL dataset.
