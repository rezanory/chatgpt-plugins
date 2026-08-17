from __future__ import annotations

import hashlib
import re

from chatgpt_plugins_core import FailureCategory, FailureReport


_RULES: list[tuple[FailureCategory, bool, float, re.Pattern[str]]] = [
    (FailureCategory.OOM, True, 0.98, re.compile(r"out of memory|cuda oom|cudnn.*alloc", re.I)),
    (FailureCategory.API_RATE_LIMIT, True, 0.98, re.compile(r"rate.?limit|too many requests|http\s*429", re.I)),
    (FailureCategory.PROVIDER_QUOTA, False, 0.98, re.compile(r"quota|usage limit|accelerator.*unavailable", re.I)),
    (FailureCategory.AUTHENTICATION, False, 0.99, re.compile(r"unauthorized|forbidden|invalid.*token|authentication", re.I)),
    (FailureCategory.DJANGO_MIGRATION, False, 0.92, re.compile(r"migration|makemigrations|migrate.*failed", re.I)),
    (FailureCategory.DJANGO_CHECK, False, 0.95, re.compile(r"systemcheckerror|manage\.py check", re.I)),
    (FailureCategory.IMPORT_ERROR, False, 0.98, re.compile(r"modulenotfounderror|importerror", re.I)),
    (FailureCategory.DEPENDENCY, False, 0.96, re.compile(r"no matching distribution|resolutionimpossible|pip.*error", re.I)),
    (FailureCategory.SYNTAX, False, 0.99, re.compile(r"syntaxerror|indentationerror", re.I)),
    (FailureCategory.TEST_FAILURE, False, 0.85, re.compile(r"pytest|\bfailed\b.*\btest", re.I)),
    (FailureCategory.TIMEOUT, True, 0.95, re.compile(r"timed out|timeout|deadline exceeded", re.I)),
    (FailureCategory.NETWORK, True, 0.94, re.compile(r"connection reset|temporary failure|dns|network is unreachable", re.I)),
    (FailureCategory.TRANSPORT, True, 0.90, re.compile(r"kernel push error|api request failed|http\s*5\d\d", re.I)),
    (FailureCategory.INFRASTRUCTURE, True, 0.90, re.compile(r"internal server error|service unavailable|runner lost", re.I)),
    (FailureCategory.POLICY, False, 0.95, re.compile(r"policy violation|not permitted|terms of service", re.I)),
]


def failure_fingerprint(category: FailureCategory, summary: str, log_excerpt: str = "") -> str:
    # Normalize volatile addresses/line numbers/timestamps while preserving the causal shape.
    text = f"{summary}\n{log_excerpt}".lower()
    text = re.sub(r"0x[0-9a-f]+", "0x#", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}[t ][0-9:.+\-z]+\b", "<time>", text)
    text = re.sub(r"line\s+\d+", "line #", text)
    text = re.sub(r"\b\d{4,}\b", "#", text)
    text = re.sub(r"\s+", " ", text).strip()[-12000:]
    payload = f"{category.value}\n{text}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def classify_failure(summary: str, log_excerpt: str = "") -> FailureReport:
    text = f"{summary}\n{log_excerpt}"[-30_000:]
    category = FailureCategory.UNKNOWN
    retryable = False
    confidence = 0.50
    for candidate, candidate_retryable, candidate_confidence, pattern in _RULES:
        if pattern.search(text):
            category = candidate
            retryable = candidate_retryable
            confidence = candidate_confidence
            break
    clean_summary = summary[:1000]
    clean_excerpt = log_excerpt[-12_000:]
    return FailureReport(
        category=category,
        summary=clean_summary,
        log_excerpt=clean_excerpt,
        retryable=retryable,
        confidence=confidence,
        fingerprint=failure_fingerprint(category, clean_summary, clean_excerpt),
    )
