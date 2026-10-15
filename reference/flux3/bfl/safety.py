"""Prompt and image checks (VLM yes/no + NSFW classifier) and EXIF provenance tags."""

import math

import torch
from PIL import ExifTags, Image
from transformers import pipeline

from flux3.text_encoder import TextEncoder

NSFW_THRESHOLD = 0.85
CHECK_AREA = 512 * 512

SYSTEM_PROMPT_CONTENT_FILTER = (
    "Your task is identifying images and text for copyright concerns and depictions of public personas."
)

PROMPT_IMAGE_INTEGRITY = """
Task: Analyze an image to identify potential copyright concerns or depictions of public figures.

Output: Respond with only "yes" or "no"

Criteria for "yes":
- The image contains a recognizable character from copyrighted media (movies, TV, comics, games, etc.)
- The image displays a trademarked logo or brand
- The image depicts a recognizable public figure (celebrities, politicians, athletes, influencers, historical figures, etc.)

Criteria for "no":
- All other cases
- When you cannot identify the specific copyrighted work or named individual

Critical Requirements:
1. You must be able to name the exact copyrighted work or specific person depicted
2. General references to demographics or characteristics are not sufficient
3. Base your decision solely on visual content, not interpretation
4. Provide only the one-word answer: "yes" or "no"
""".strip()

PROMPT_IMAGE_INTEGRITY_FOLLOW_UP = "Does this image have copyright concerns or includes public figures?"

PROMPT_TEXT_INTEGRITY = """
Task: Analyze a text prompt to identify potential copyright concerns or requests to depict living public figures.

Output: Respond with only "yes" or "no"

Criteria for "Yes":
- The prompt explicitly names a character from copyrighted media (movies, TV, comics, games, etc.)
- The prompt explicitly mentions a trademarked logo or brand
- The prompt names or describes a specific living public figure (celebrities, politicians, athletes, influencers, etc.)

Criteria for "No":
- All other cases
- When you cannot identify the specific copyrighted work or named individual

Critical Requirements:
1. You must be able to name the exact copyrighted work or specific person referenced
2. General demographic descriptions or characteristics are not sufficient
3. Analyze only the prompt text, not potential image outcomes
4. Provide only the one-word answer: "yes" or "no"

The prompt to check is:
-----
{prompt}
-----

Does this prompt have copyright concerns or includes public figures?
""".strip()


class Safety:
    def __init__(self, text_encoder: TextEncoder, classifier: str, upsampler=None):
        self.model, self.processor = text_encoder.model, text_encoder.processor
        self.upsampler = upsampler
        self.nsfw = pipeline("image-classification", model=classifier, device=text_encoder.device)
        tokenizer = upsampler.tokenizer if upsampler else self.processor.tokenizer
        self.yes, self.no = (tokenizer.encode(word, add_special_tokens=False)[0] for word in ("yes", "no"))

    def prompt_flagged(self, prompt: str) -> bool:
        return self._says_yes([{"type": "text", "text": PROMPT_TEXT_INTEGRITY.format(prompt=prompt)}])

    def image_flagged(self, image: Image.Image) -> str:
        """Empty if the image passes, else which check flagged it."""
        image = image.convert("RGB")
        if next(c for c in self.nsfw(image) if c["label"] == "nsfw")["score"] > NSFW_THRESHOLD:
            return "nudity or other adult content"
        scale = (CHECK_AREA / (image.width * image.height)) ** 0.5
        image = image.resize((max(1, int(scale * image.width)), max(1, int(scale * image.height))))
        flagged = self._says_yes([{"type": "text", "text": PROMPT_IMAGE_INTEGRITY}, {"type": "image", "image": image},
                                  {"type": "text", "text": PROMPT_IMAGE_INTEGRITY_FOLLOW_UP}])
        return "a copyrighted character, a brand or a public figure" if flagged else ""

    def _says_yes(self, content: list[dict]) -> bool:
        """The next token restricted to yes / no."""
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_CONTENT_FILTER}]},
                    {"role": "user", "content": content}]
        return self._vllm(messages) if self.upsampler else self._text_encoder(messages)

    def _vllm(self, messages: list[dict]) -> bool:
        from vllm import SamplingParams

        images = [part.pop("image") for message in messages if isinstance(message["content"], list)
                  for part in message["content"] if part["type"] == "image"]
        text = self.upsampler.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                             enable_thinking=False)
        request = {"prompt": text, **({"multi_modal_data": {"image": images}} if images else {})}
        params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=2, allowed_token_ids=[self.yes, self.no])
        logprobs = self.upsampler.llm.generate([request], params, use_tqdm=False)[0].outputs[0].logprobs[0]
        yes, no = (logprobs[t].logprob if t in logprobs else -math.inf for t in (self.yes, self.no))
        return yes > no

    @torch.inference_mode()
    def _text_encoder(self, messages: list[dict]) -> bool:
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        logits = self.model(**inputs, use_cache=False).logits[0, -1]
        return bool(logits[self.yes] > logits[self.no])


def exif(model: str) -> Image.Exif:
    tags = Image.Exif()
    tags[ExifTags.Base.Software] = f"AI generated;{model}"
    tags[ExifTags.Base.Make] = "Black Forest Labs"
    return tags
