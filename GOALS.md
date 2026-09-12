# Long Term Vision + Goal

This project explores the most efficient ways to run a variets of AI/LLM generations on **Vulkan** using the AMD AI MAX+ 395 "Strix Halo" platform, with gfx1151 and RDNA3.5. 

Ultimately we want to be able to run these tasks:
  1. Speech to Text via Nvidia's `parakeet-tdt-0.6b-v3`:
    * https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
    * https://arxiv.org/html/2509.14128v2
  2. Text generation via Qwen's `Qwen3.8-Next-Flash`:
    * https://huggingface.co/Qwen/Qwen3.8-Flash-Next
    * https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf
    * We'll need to quantise the weights once we've determined the most efficient method for the hardware
  3. Text to Speech via Hexgrad's `kokoro-82M`:
    * https://huggingface.co/hexgrad/Kokoro-82M
    * https://arxiv.org/html/2306.07691v2
  4. Image generation via Tongyi's `z-image-turbo`:
    * https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
    * https://arxiv.org/html/2511.22699v5

We will ultimately serve up a HTTP API serving these features, in Go.
