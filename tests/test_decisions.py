# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI Decisions API readouts behind /v1/decisions."""

import asyncio
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pydantic import ValidationError  # noqa: E402

from omlx.api import decisions  # noqa: E402
from omlx.api.decisions_models import DecisionRequest  # noqa: E402

VOCAB = 256


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class _FakeChatEngine:
    """Chat engine whose next-token logits come from ``scores(prompt)``."""

    def __init__(self, scores):
        self.tokenizer = _CharTokenizer()
        self.scores = scores
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        content = messages[-1]["content"]
        text = (
            content
            if isinstance(content, str)
            else "".join(p.get("text", "") for p in content)
        )
        letters = re.findall(r"^\[([A-Z])\]", text, flags=re.M)
        logits = np.full(VOCAB, -5.0, dtype=np.float32)
        for letter, value in zip(letters, self.scores(text, letters), strict=True):
            logits[ord(letter)] = value
        kwargs["logits_observer"](None, mx.array(logits)[None])
        return SimpleNamespace(prompt_tokens=len(text))


def _request(**body):
    return DecisionRequest.model_validate(
        {"model": "m", "input": "Checkout is down.", **body}
    )


def _run(engine, request, multimodal=False):
    return asyncio.run(decisions.decide(engine, request, multimodal))


def test_capture_reads_option_logprobs_and_keeps_logits():
    capture = decisions.OptionLogitsCapture([[1, 2], [3]])
    logits = mx.array([[0.0, 1.0, 1.0, 2.0]])
    out = capture(None, logits)
    assert out is logits
    total = math.log(sum(math.exp(v) for v in (0.0, 1.0, 1.0, 2.0)))
    assert capture.logprobs == pytest.approx(
        [math.log(2 * math.e) - total, 2.0 - total], rel=1e-5
    )
    capture(None, mx.array([[9.0, 0.0, 0.0, 0.0]]))
    assert capture.logprobs[1] == pytest.approx(2.0 - total, rel=1e-5)


def test_letter_ids_merge_spellings_and_reject_collisions():
    class Spaced:
        def encode(self, text, add_special_tokens=False):
            return [ord(text.strip()) + (1000 if text.startswith(" ") else 0)]

    ids = decisions.letter_token_ids(Spaced())
    assert ids[0] == [ord("A"), 1000 + ord("A")]

    class Collapsed:
        def encode(self, text, add_special_tokens=False):
            return [7]

    with pytest.raises(decisions.DecisionsRequestError):
        decisions.letter_token_ids(Collapsed())


def test_choice_answer_probabilities_and_confidence():
    engine = _FakeChatEngine(lambda text, letters: [2.0, 0.0, math.log(0.5)])
    request = _request(
        questions=[
            {
                "type": "choice",
                "name": "route",
                "instructions": "Which team?",
                "choices": [
                    {"value": "billing", "description": "Payments"},
                    "account",
                    {"value": "other"},
                ],
            }
        ]
    )
    result = _run(engine, request)
    (answer,) = result["answers"]
    weights = np.exp([2.0, 0.0, math.log(0.5)])
    probs = weights / weights.sum()
    assert answer["type"] == "choice" and answer["name"] == "route"
    assert answer["choice"] == "billing"
    assert [p["value"] for p in answer["probabilities"]] == [
        "billing",
        "account",
        "other",
    ]
    assert [p["probability"] for p in answer["probabilities"]] == pytest.approx(
        probs, abs=1e-4
    )
    assert answer["confidence"] == pytest.approx((probs[0] - 1 / 3) / (2 / 3), abs=1e-4)
    text = engine.calls[0]["messages"][0]["content"]
    assert text.startswith("Checkout is down.\n\nQuestion: Which team?")
    assert "[A] billing: Payments\n[B] account\n[C] other" in text
    assert engine.calls[0]["max_tokens"] == 1
    assert engine.calls[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_predicate_and_score_answers():
    def scores(text, letters):
        if "Yes" in text:
            return [math.log(0.8), math.log(0.2)]
        return [math.log(0.1), math.log(0.7), math.log(0.2)]

    request = _request(
        questions=[
            {"type": "predicate", "name": "urgent", "instructions": "Is it urgent?"},
            {
                "type": "score",
                "name": "severity",
                "instructions": "How severe?",
                "levels": [
                    {"label": "low"},
                    {"label": "medium", "description": "Some users"},
                    "high",
                ],
            },
        ]
    )
    result = _run(_FakeChatEngine(scores), request)
    predicate, score = result["answers"]
    assert predicate == {"type": "predicate", "name": "urgent", "probability": 0.8}
    assert score["score"] == pytest.approx(1.1, abs=1e-4)
    assert score["probabilities"] == [
        {"value": 0, "label": "low", "probability": 0.1},
        {"value": 1, "label": "medium", "probability": 0.7},
        {"value": 2, "label": "high", "probability": 0.2},
    ]
    assert 0.0 <= score["confidence"] <= 1.0
    assert result["input_tokens"] > 0


def test_every_readout_shares_the_input_prefix():
    engine = _FakeChatEngine(lambda text, letters: [0.0] * len(letters))
    request = _request(
        questions=[
            {"type": "predicate", "name": f"q{i}", "instructions": f"Q{i}?"}
            for i in range(5)
        ]
    )
    _run(engine, request)
    prompts = [c["messages"][0]["content"] for c in engine.calls]
    assert len(prompts) == 5
    assert all(p.startswith("Checkout is down.\n\nQuestion: ") for p in prompts)


def test_many_choices_compose_group_readouts():
    count = 30
    target = np.linspace(0.0, 3.0, count)

    def scores(text, letters):
        values = re.findall(r"^\[[A-Z]\] v(\d+)$", text, flags=re.M)
        return [target[int(v)] for v in values]

    request = _request(
        questions=[
            {
                "type": "choice",
                "name": "pick",
                "instructions": "Pick one",
                "choices": [f"v{i}" for i in range(count)],
            }
        ]
    )
    engine = _FakeChatEngine(scores)
    (answer,) = _run(engine, request)["answers"]
    # Two groups of 15 plus the readout over the two winners.
    assert len(engine.calls) == 3
    probs = [p["probability"] for p in answer["probabilities"]]
    full = np.exp(target - target.max())
    assert probs == pytest.approx(full / full.sum(), abs=1e-4)
    assert answer["choice"] == f"v{count - 1}"


def test_masked_option_letters_are_a_refusal():
    engine = _FakeChatEngine(lambda text, letters: [-np.inf] * len(letters))
    request = _request(
        questions=[{"type": "predicate", "name": "p", "instructions": "Is it?"}]
    )
    assert _run(engine, request)["answers"] == [{"type": "refusal", "name": "p"}]


def test_message_input_with_images():
    image = "data:image/png;base64,AAAA"
    request = DecisionRequest.model_validate(
        {
            "model": "m",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Look:"},
                        {"type": "input_image", "image_url": image},
                    ],
                }
            ],
            "questions": [{"type": "predicate", "name": "p", "instructions": "Cat?"}],
        }
    )
    engine = _FakeChatEngine(lambda text, letters: [0.0, 0.0])
    _run(engine, request, multimodal=True)
    content = engine.calls[0]["messages"][0]["content"]
    assert content[:2] == [
        {"type": "text", "text": "Look:"},
        {"type": "image_url", "image_url": {"url": image}},
    ]
    assert content[2]["text"].startswith("\n\nQuestion: Cat?")

    with pytest.raises(decisions.DecisionsRequestError):
        _run(engine, request, multimodal=False)


def test_engine_without_logits_is_rejected():
    class NoLogits(_FakeChatEngine):
        async def chat(self, messages, **kwargs):
            return SimpleNamespace(prompt_tokens=1)

    request = _request(
        questions=[{"type": "predicate", "name": "p", "instructions": "Is it?"}]
    )
    with pytest.raises(decisions.DecisionsRequestError):
        _run(NoLogits(None), request)


@pytest.mark.parametrize(
    "change",
    [
        {"questions": []},
        {"questions": [{"type": "rank", "name": "a", "instructions": "x"}]},
        {"questions": [{"type": "predicate", "name": "a", "instructions": "x"}] * 2},
        {
            "questions": [
                {"type": "choice", "name": "a", "instructions": "x", "choices": ["y"]}
            ]
        },
        {"input": [{"content": [{"type": "input_image", "image_url": "https://x"}]}]},
    ],
)
def test_schema_validation(change):
    body = {
        "model": "m",
        "input": "x",
        "questions": [{"type": "predicate", "name": "a", "instructions": "x"}],
        **change,
    }
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(body)


def test_systemone_translation_round_trip():
    request = _request(
        questions=[
            {"type": "predicate", "name": "p", "instructions": "Is it?"},
            {
                "type": "choice",
                "name": "c",
                "instructions": "Which?",
                "choices": [{"value": "x", "description": "X"}, "y"],
            },
            {
                "type": "score",
                "name": "s",
                "instructions": "How much?",
                "levels": ["low", {"label": "high", "description": "a lot"}],
            },
        ]
    )
    plan = decisions.to_systemone(request)
    assert plan["state"] == "Checkout is down."
    assert plan["questions"] == {
        "p": {"type": "noul", "instructions": "Is it?", "criteria": None},
        "c": {
            "type": "choice",
            "instructions": "Which?",
            "criteria": {"x": "X", "y": None},
        },
        "s": {
            "type": "score",
            "instructions": "How much?",
            "criteria": ["low", "high: a lot"],
        },
    }
    answers = decisions.from_systemone(
        request,
        {
            "p": {"type": "noul", "noul": 0.3},
            "c": {
                "type": "choice",
                "choice": "y",
                "probabilities": {"x": 0.4, "y": 0.6},
                "confidence": 0.2,
            },
            "s": {
                "type": "score",
                "score": 0.9,
                "probabilities": {"0": 0.1, "1": 0.9},
                "confidence": 0.8,
            },
        },
    )
    assert answers == [
        {"type": "predicate", "name": "p", "probability": 0.3},
        {
            "type": "choice",
            "name": "c",
            "choice": "y",
            "probabilities": [
                {"value": "x", "probability": 0.4},
                {"value": "y", "probability": 0.6},
            ],
            "confidence": 0.2,
        },
        {
            "type": "score",
            "name": "s",
            "score": 0.9,
            "probabilities": [
                {"value": 0, "label": "low", "probability": 0.1},
                {"value": 1, "label": "high", "probability": 0.9},
            ],
            "confidence": 0.8,
        },
    ]
