# Long Term Vision + Goal

This project explores the most efficient ways to run a variety of AI/LLM generations on **Vulkan** using the AMD AI MAX+ 395 "Strix Halo" platform, with gfx1151 and RDNA3.5. 

Ultimately we want to be able to run these tasks:
  1. Speech to Text via Nvidia's `parakeet-tdt-0.6b-v3`:
    * https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
    * https://arxiv.org/html/2509.14128v2
  2. Text generation via Qwen's `Qwen3.8-Next-Flash`:
    * https://huggingface.co/Qwen/Qwen3.8-Flash-Next
    * https://arxiv.org/html/2608.30320v1
    * We'll need to quantise the weights once we've determined the most efficient method for the hardware
  3. Text to Speech via Hexgrad's `kokoro-82M`:
    * https://huggingface.co/hexgrad/Kokoro-82M
    * https://arxiv.org/html/2306.07691v2
  4. Image generation and editing via Qwen's `Qwen-Image-2.1`:
    * https://huggingface.co/Qwen/Qwen-Image-2.1
    * https://qwen.ai/blog?id=qwen-image-2.1
    * Unified text-to-image and reference-image editing (up to 10 refs), native RGBA transparency
    * In-progress previews: no tiny autoencoder exists yet for its 64-channel VAE,
      so we fit our own linear preview and watch https://github.com/madebyollin/taehv
      for a proper one (replaced z-image-turbo + taef1 on 2026-09-20;
      the vertical is served and parked — research/qimage-vertical.md)
  5. Compute embeddings via Qwen's `Qwen3-Embedding-0.6B`:
    * https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
    * https://arxiv.org/html/2506.05176v3
  6. Classification via `kev-4b`:
    * https://huggingface.co/jaredpalmer/kev-4b
    * https://github.com/jaredpalmer/kev
    * https://archerhume.com/posts/jevs-architecture-unmasked
  7. Video generation via MiniMax's `H3`:
    * https://huggingface.co/MiniMaxAI/MiniMax-H3
    * https://github.com/MiniMax-AI/MiniMax-H3

We will ultimately serve up a HTTP API serving these features, in Go.
