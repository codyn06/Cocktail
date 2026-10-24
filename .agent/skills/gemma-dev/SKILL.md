---
name: gemma-dev
description: Use when writing, running, or debugging code that uses Google's Gemma open models (Gemma 4 / 3n) - hosted through the Gemini API (google-genai), local inference (Hugging Face Transformers, Ollama, llama.cpp), or on-device (LiteRT-LM). Covers model choice, audio/image input limits, chat, thinking, function calling, and verification of model IDs and hardware.
---

# Gemma Development

Project-authored skill, not an official vendor skill. Written 2026-10-24 from
https://ai.google.dev/gemma/docs/get_started and linked pages. Those pages say they are
deprecated in favor of https://gemma.ai, so check gemma.ai first and treat
everything below as a starting point to verify, not as settled fact.

## Do not assume

- That a variant or runtime has been chosen. Neither has (`.agents/decisions.md`).
- That model weights are downloaded or accepted, or that a GPU/phone can run a variant.
  Check memory and latency on the target hardware and label the evidence.
- That a model ID, parameter, or feature exists until a current official page or a call
  that ran confirms it.

## Choose the variant (Gemma 4)

| Variant | Inputs | Context | Intended platform |
|---|---|---|---|
| E2B | text, image, audio | 128K | phones |
| E4B | text, image, audio | 128K | phones, laptops |
| 12B Unified | text, image, audio | 256K | laptops, desktops, small servers |
| 26B A4B (MoE) | text, image | 256K | desktops, small servers |
| 31B Dense | text, image | 256K | large servers |

All variants output text only. Cocktail is audio-first, so only E2B, E4B, or 12B can
take audio directly. The larger models need a separate speech-to-text step. Gemma 3n
(E2B/E4B) also accepts audio.

Model card: https://ai.google.dev/gemma/docs/core/model_card_4

## Input limits

- **Audio**: up to 30 s per clip, mono, 16 kHz, float32 in [-1, 1], 25 tokens/s for
  Gemma 4. Split longer audio into chunks. Downmix stereo, resample, and decode WebM
  or MP3 (e.g. with ffmpeg) before use. https://ai.google.dev/gemma/docs/capabilities/audio
- **Video**: frames only, up to about 60 s at 1 fps.
- **Images**: visual token budget of 70, 140, 280, 560, or 1120 per image. Lower budgets
  are faster but lose detail.

## Runtimes

| Runtime | When | Docs |
|---|---|---|
| Gemini API (hosted) | fastest start, no GPU; needs an API key and network | https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api |
| HF Transformers | full feature set (audio, tools) on a GPU box | https://ai.google.dev/gemma/docs/core/huggingface_inference |
| Ollama / llama.cpp / LM Studio | local quantized server | https://ai.google.dev/gemma/docs/integrations/ollama |
| LiteRT-LM | on-device (phone) | https://ai.google.dev/edge/litert-lm/overview |
| Vertex AI / GKE | managed or self-hosted cloud | https://ai.google.dev/gemma/docs/integrations/google-cloud |

Weights: https://huggingface.co/collections/google/gemma-4 and
https://www.kaggle.com/models/google/gemma-4. Gemma 4 is Apache-2.0. Downloading
weights still needs user approval (AGENTS.md).

## Hosted: Gemini API

Documented IDs: `gemma-4-31b-it` and `gemma-4-26b-a4b-it`. The hosted page does not
list audio support, so test before relying on audio here. Use the `google-genai` SDK
(Python) or `@google/genai` (JS). The key comes from the `GEMINI_API_KEY` env var. Never
commit it.

```python
from google import genai
from google.genai import types

client = genai.Client()
response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents="Describe the scene in one sentence.",
    config=types.GenerateContentConfig(
        system_instruction="You are a concise audio-first assistant.",
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),  # "high" to enable
    ),
)
print(response.text)
```

```javascript
import { GoogleGenAI } from "@google/genai";
const ai = new GoogleGenAI();
const response = await ai.models.generateContent({
  model: "gemma-4-26b-a4b-it",
  contents: "Describe the scene in one sentence.",
});
console.log(response.text);
```

Also documented: images through `client.files.upload(...)` in `contents`, multi-turn
chat with `client.chats.create(model=...)` and `chat.send_message(...)`, function
declarations in `tools` (results in `response.function_calls`), and Google Search
grounding with `tools=[{"google_search": {}}]`. Rate limits and pricing are not on the
Gemma page. Check the Gemini API pricing page.

## Local: Transformers (audio example)

```python
from transformers import pipeline, GenerationConfig

MODEL_ID = "google/gemma-4-E4B-it"  # or google/gemma-4-E2B-it, google/gemma-4-12B-it
pipe = pipeline(task="any-to-any", model=MODEL_ID, device_map="auto", dtype="auto")
config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 64

messages = [{"role": "user", "content": [
    {"type": "text", "text": "Transcribe the following speech segment in its original language."},
    {"type": "audio", "audio": "clip_16k_mono.wav"},
]}]
out = pipe(messages, return_full_text=False, generate_kwargs=dict(generation_config=config))
print(out[0]["generated_text"])  # output may end with a `<turn|>` token you need to strip
```

## Function calling (local, Gemma 4)

Docs: https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4.
Prompt format: https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4.

- Pass `tools` (JSON schema or Python functions with Google-style docstrings) to
  `processor.apply_chat_template(...)`.
- The model only returns a call, e.g. `<|tool_call>call:name{arg:<|"|>v<|"|>}<tool_call|>`.
  Your code parses it, runs the function, appends `tool_calls` and `tool_responses`, and
  applies the template again to get the final answer.
- Dispatch through an explicit allow-list and validate the arguments. Never use
  `globals()`. Regex parsing breaks on nested arguments, and auto-generated schemas
  flatten nested objects, so write the schema by hand for complex arguments.
- `enable_thinking=True` with `processor.parse_response()` splits thoughts, answer,
  and tool calls.

## Verify before claiming it works

- Record the model ID, runtime and version, hardware, and the measured latency for each
  result. Label the evidence as hardware-tested, mocked, replayed, or untested.
- For audio, confirm the clip you send is 16 kHz mono and 30 s or shorter. A wrong
  format can still return plausible-looking text.
- Other references: thinking https://ai.google.dev/gemma/docs/capabilities/thinking,
  images https://ai.google.dev/gemma/docs/capabilities/vision/image, releases
  https://ai.google.dev/gemma/docs/releases.
