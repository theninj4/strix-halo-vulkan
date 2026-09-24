"""Render the qwen3.8-flash-next chat template with Jinja, as the oracle for
llm.RenderChat (LLM.md L9a).

`llm/chat.go` is a transcription of `tokenizer.chat_template` -- 180 lines of
Jinja living inside the GGUF's metadata -- and a transcription is only worth
anything if something checks it. This renders the original with the same
environment transformers builds (`ImmutableSandboxedEnvironment`, trim_blocks,
lstrip_blocks, loopcontrols, and a `tojson` bound to
`json.dumps(ensure_ascii=False)`), over a corpus chosen for the branches the
template actually has: the merged system turn, each reasoning effort, thinking
on and off, the tool block, an assistant turn with calls, and a run of tool
results.

The template itself is not in this repository -- it is checkpoint metadata --
so export it first:

    go run ./cmd/llm -model models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
        -chat-template > reference/out/chat/template.jinja
    .venv/bin/python reference/dump_chat_template.py
"""

import argparse
import json
import os

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in a city, in °C",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "e.g. Wimbledon"},
                "days": {"type": "integer", "description": "how far ahead"},
                "units": {"type": ["string", "null"]},
            },
            "required": ["city"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": 'Search the web for "documents" & things',
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": "object"},
                "limit": {"type": "number"},
                "recent": {"type": "boolean"},
            },
        },
    },
}

U = "user"
A = "assistant"
S = "system"

# Every case is (name, messages, tools, kwargs). kwargs are the template's own
# switches; add_generation_prompt is on unless a case says otherwise.
CASES = [
    ("user_only", [{"role": U, "content": "What is the capital of France?"}], None, {}),
    ("user_untrimmed", [{"role": U, "content": "  \n What is 2 + 2?  \n\n "}], None, {}),
    ("system_user", [
        {"role": S, "content": "You are terse."},
        {"role": U, "content": "Hello."},
    ], None, {}),
    ("two_systems", [
        {"role": S, "content": "You are terse."},
        {"role": S, "content": "  You answer in English.  "},
        {"role": U, "content": "Hello."},
    ], None, {}),
    ("developer_role", [
        {"role": "developer", "content": "You are a compiler."},
        {"role": U, "content": "int main;"},
    ], None, {}),
    ("empty_system", [
        {"role": S, "content": "   "},
        {"role": U, "content": "Hello."},
    ], None, {}),
    ("effort_low", [{"role": U, "content": "Hello."}], None, {"reasoning_effort": "low"}),
    ("effort_medium", [{"role": U, "content": "Hello."}], None, {"reasoning_effort": "medium"}),
    ("effort_high", [{"role": U, "content": "Hello."}], None, {"reasoning_effort": "high"}),
    ("effort_xhigh", [{"role": U, "content": "Hello."}], None, {"reasoning_effort": "xhigh"}),
    ("no_thinking", [
        {"role": S, "content": "You are terse."},
        {"role": U, "content": "Hello."},
    ], None, {"enable_thinking": False}),
    ("no_generation_prompt", [
        {"role": U, "content": "Hello."},
        {"role": A, "content": "Hi.", "reasoning_content": "They said hello."},
    ], None, {"add_generation_prompt": False}),
    ("multi_turn", [
        {"role": S, "content": "You are terse."},
        {"role": U, "content": "What is 2 + 2?"},
        {"role": A, "content": "4.", "reasoning_content": "  Two and two.  "},
        {"role": U, "content": "And 3 + 3?"},
    ], None, {}),
    ("multi_turn_no_reasoning", [
        {"role": U, "content": "What is 2 + 2?"},
        {"role": A, "content": "4."},
        {"role": U, "content": "And 3 + 3?"},
    ], None, {}),
    ("drop_thinking", [
        {"role": U, "content": "What is 2 + 2?"},
        {"role": A, "content": "4.", "reasoning_content": "Two and two."},
        {"role": U, "content": "And 3 + 3?"},
        {"role": A, "content": "6.", "reasoning_content": "Three and three."},
    ], None, {"preserve_thinking": False}),
    ("tools_only", [{"role": U, "content": "Weather in Wimbledon?"}], [WEATHER_TOOL], {}),
    ("tools_and_system", [
        {"role": S, "content": "You are terse."},
        {"role": U, "content": "Weather in Wimbledon?"},
    ], [WEATHER_TOOL, SEARCH_TOOL], {}),
    ("tools_effort_low", [
        {"role": U, "content": "Weather in Wimbledon?"},
    ], [WEATHER_TOOL], {"reasoning_effort": "low"}),
    ("tool_call", [
        {"role": U, "content": "Weather in Wimbledon?"},
        {"role": A, "content": "", "reasoning_content": "I should call the tool.",
         "tool_calls": [{"type": "function", "function": {
             "name": "get_weather",
             "arguments": {"city": "Wimbledon", "days": 2, "units": None}}}]},
        {"role": "tool", "content": "18 °C, cloudy"},
    ], [WEATHER_TOOL], {}),
    ("tool_call_with_text", [
        {"role": U, "content": "Weather in Wimbledon, and search for tennis?"},
        {"role": A, "content": "Let me look both up.", "reasoning_content": "Two calls.",
         "tool_calls": [
             {"type": "function", "function": {
                 "name": "get_weather", "arguments": {"city": "Wimbledon", "days": 1}}},
             {"type": "function", "function": {
                 "name": "search", "arguments": {
                     "query": "tennis \"grass\"", "filters": {"since": 2020, "lang": "en"},
                     "limit": 3.5, "recent": True}}},
         ]},
        {"role": "tool", "content": "18 °C, cloudy"},
        {"role": "tool", "content": "Wimbledon is played on grass."},
        {"role": A, "content": "Cloudy, 18, and it is on grass.", "reasoning_content": "Both came back."},
        {"role": U, "content": "Thanks."},
    ], [WEATHER_TOOL, SEARCH_TOOL], {}),
    ("multiline_argument", [
        {"role": U, "content": "Search for this."},
        {"role": A, "content": "", "tool_calls": [{"type": "function", "function": {
            "name": "search", "arguments": {"query": "line one\nline two"}}}]},
    ], [SEARCH_TOOL], {"add_generation_prompt": False}),
    ("unicode", [
        {"role": S, "content": "Tu es concis. 你很简洁。"},
        {"role": U, "content": "Café ou thé ? 🍵"},
    ], None, {}),
    ("tool_response_last_query", [
        # The last *user* turn is a tool response written by hand, so the
        # template's last_query_index falls back past it.
        {"role": U, "content": "Weather?"},
        {"role": A, "content": "", "reasoning_content": "Call it.",
         "tool_calls": [{"type": "function", "function": {
             "name": "get_weather", "arguments": {"city": "Wimbledon"}}}]},
        {"role": U, "content": "<tool_response>\n18 °C\n</tool_response>"},
        {"role": A, "content": "18 degrees.", "reasoning_content": "It came back."},
    ], [WEATHER_TOOL], {"preserve_thinking": False, "add_generation_prompt": False}),
    # LLM-VISION.md V7: content lists with images in them.
    ("image_then_text", [
        {"role": U, "content": [{"type": "image"}, {"type": "text", "text": "What is in this picture?"}]},
    ], None, {}),
    ("images_between_text", [
        {"role": U, "content": [
            {"type": "text", "text": "  Compare "}, {"type": "image"},
            {"type": "text", "text": " with "}, {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "text", "text": ".  \n"}]},
    ], None, {}),
    ("vision_ids_across_turns", [
        {"role": S, "content": "Describe pictures."},
        {"role": U, "content": [{"type": "image"}, {"type": "text", "text": "This one?"}]},
        {"role": A, "content": "A barn.", "reasoning_content": "Look."},
        {"role": U, "content": [{"type": "text", "text": "And these: "}, {"type": "image"}, {"type": "image"}]},
    ], None, {"add_vision_id": True}),
    ("image_in_tool_response", [
        {"role": U, "content": "Take a screenshot."},
        {"role": A, "content": "", "reasoning_content": "Call it.",
         "tool_calls": [{"type": "function", "function": {
             "name": "search", "arguments": {"query": "screen"}}}]},
        {"role": U, "content": [
            {"type": "text", "text": "<tool_response>\n"}, {"type": "image"},
            {"type": "text", "text": "\n</tool_response>"}]},
    ], [SEARCH_TOOL], {"preserve_thinking": False, "add_vision_id": True}),
    ("image_no_thinking", [
        {"role": U, "content": [{"type": "image"}, {"type": "text", "text": "Caption?"}]},
    ], None, {"enable_thinking": False}),
]


def build_env():
    """The environment transformers builds for a chat template.

    The two things that matter are `trim_blocks`/`lstrip_blocks`, which decide
    what the whitespace between tags renders as, and `tojson` -- Jinja's own
    escapes HTML, so transformers replaces it with json.dumps(ensure_ascii=
    False), whose separators carry spaces that Go's encoder does not write.
    """

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def go_case(name, messages, tools, kwargs, rendered):
    """The same case in the shape llm.ChatMessage and llm.ChatOpts read."""
    out_msgs = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            gm = {"role": m["role"], "content": "", "parts": [
                {"image": True} if (it.get("type") in ("image", "image_url")) else {"text": it["text"]}
                for it in content]}
        else:
            gm = {"role": m["role"], "content": content or ""}
        if m.get("reasoning_content"):
            gm["reasoning"] = m["reasoning_content"]
        calls = []
        for c in m.get("tool_calls", []):
            fn = c.get("function", c)
            calls.append({
                "name": fn["name"],
                "args": [{"name": k, "value": v} for k, v in fn.get("arguments", {}).items()],
            })
        if calls:
            gm["tool_calls"] = calls
        out_msgs.append(gm)
    opts = {}
    if kwargs.get("reasoning_effort"):
        opts["effort"] = kwargs["reasoning_effort"]
    if kwargs.get("enable_thinking") is False:
        opts["no_thinking"] = True
    if kwargs.get("preserve_thinking") is False:
        opts["drop_thinking"] = True
    if kwargs.get("add_generation_prompt") is False:
        opts["no_generation_prompt"] = True
    if kwargs.get("add_vision_id"):
        opts["add_vision_id"] = True
    return {
        "name": name,
        "messages": out_msgs,
        "tools": tools or [],
        "opts": opts,
        "rendered": rendered,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="reference/out/chat/template.jinja")
    ap.add_argument("--out", default="reference/out/chat")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.template) as fh:
        template = build_env().from_string(fh.read())

    cases = []
    for name, messages, tools, kwargs in CASES:
        kw = dict(kwargs)
        kw.setdefault("add_generation_prompt", True)
        rendered = template.render(messages=messages, tools=tools, **kw)
        cases.append(go_case(name, messages, tools, kwargs, rendered))
        print(f"  {name:24s} {len(rendered):6d} chars")

    path = os.path.join(args.out, "cases.json")
    with open(path, "w") as fh:
        json.dump({"cases": cases}, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
