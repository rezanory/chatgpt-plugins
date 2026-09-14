#!/usr/bin/env python3
"""Conservatively classify recurring Kaggle campaign incident signals."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

MARKER = "POST failed with:"


def _canonical_not_found_envelope(text: str) -> bool:
    decoder = json.JSONDecoder()
    offset = 0
    while True:
        marker_at = text.find(MARKER, offset)
        if marker_at < 0:
            return False
        prefix = text[max(0, marker_at - 300) : marker_at]
        candidate = text[marker_at + len(MARKER) :].lstrip()
        try:
            payload, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            offset = marker_at + len(MARKER)
            continue
        code = payload.get("error", {}).get("code") if isinstance(payload, dict) else None
        if (
            "BackendError" in prefix
            and isinstance(payload, dict)
            and payload.get("wasSuccessful") is False
            and isinstance(payload.get("error"), dict)
            and type(code) is int
            and code == 5
            and payload.get("errors") == ["Not found"]
        ):
            return True
        offset = marker_at + len(MARKER)


def _finding(
    classification: str,
    subreason: str,
    confidence: str,
    next_checks: list[str],
) -> dict[str, Any]:
    return {
        "classification": classification,
        "subreason": subreason,
        "confidence": confidence,
        "next_checks": next_checks,
        "automatic_retry_allowed": False,
    }


def classify(text: str) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    lowered = text.lower()

    if "persistence_restore_failed" in lowered:
        if _canonical_not_found_envelope(text):
            findings.append(
                _finding(
                    "BLOCKED_TEST_CONTRACT_DRIFT",
                    (
                        "fresh persistence absence uses the exact KaggleHub "
                        "BackendError NOT_FOUND envelope"
                    ),
                    "exact-pattern",
                    [
                        (
                            "confirm the campaign is fresh and the persistence artifact "
                            "is expected to be absent"
                        ),
                        "test HTTP 404 and the exact code=5 envelope as positives",
                        (
                            "test permission, wrong-code, malformed, auth, and network "
                            "cases as negatives"
                        ),
                    ],
                )
            )
        else:
            findings.append(
                _finding(
                    "EVIDENCE_GAP",
                    "persistence restore failed but confirmed fresh absence is not proven",
                    "fail-closed",
                    [
                        "capture the exception type, status code, and parsed provider payload",
                        (
                            "do not convert generic code=5, permission, malformed, auth, "
                            "or network errors to absence"
                        ),
                    ],
                )
            )

    if re.search(
        r"(?:sha(?:-?256)?|hash).{0,80}(?:mismatch|does not match)", text, re.IGNORECASE
    ) or re.search(
        r"(?:mismatch|does not match).{0,80}(?:sha(?:-?256)?|hash)", text, re.IGNORECASE
    ):
        findings.append(
            _finding(
                "BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH",
                "expected and observed hash-bound payload identities differ",
                "exact-pattern",
                [
                    (
                        "compare canonical Git bytes, builder output, workflow payload, "
                        "and provider upload bytes"
                    ),
                    "check text-mode newline translation and use explicit UTF-8 byte writes",
                    "preserve any already-launched candidate and fix forward",
                ],
            )
        )

    if re.search(r"(?:http(?: error)?\s*)?403\b|forbidden", text, re.IGNORECASE):
        findings.append(
            _finding(
                "BLOCKED_ADMISSION_AUTHORITY_MISSING",
                (
                    "403 may be OIDC workflow/ref/event policy or account authorization; "
                    "the symptom alone is non-diagnostic"
                ),
                "needs-reconciliation",
                [
                    (
                        "compare OIDC workflow_ref, event_name, ref, actor, and repository "
                        "claims with deployed trust"
                    ),
                    (
                        "inspect trust-entry history before changing credentials or "
                        "re-authorizing a workflow"
                    ),
                    "probe the same account through the permanent allowlisted read path",
                ],
            )
        )

    if any(
        signal in lowered
        for signal in (
            "connection interrupted",
            "stopped thinking",
            "failed to generate",
            "response generation failed",
            "delivery timeout",
        )
    ):
        findings.append(
            _finding(
                "BLOCKED_CHAT_STREAM_INTERRUPTED_IDENTITY_DRIFT",
                "chat/UI transport ended without proving the execution state",
                "exact-pattern",
                [
                    "resolve the stable conversation and project/worktree binding",
                    "reconcile GitHub and provider side effects before continuing",
                    "resume from the unfinished point and quarantine any duplicate launch",
                ],
            )
        )

    return {
        "schema_version": 1,
        "findings": findings,
        "deterministic_finding_count": len(findings),
        "fail_closed": True,
        "automatic_retry_allowed": False,
        "note": (
            (
                "No known deterministic signal found; collect exact chat, GitHub, "
                "provider, and identity evidence."
            )
            if not findings
            else (
                "Findings are diagnostic only and do not authorize source, policy, "
                "credential, or compute mutations."
            )
        ),
    }


def _self_test() -> None:
    exact = (
        'BackendError: POST failed with: {"errors":["Not found"],'
        '"error":{"code":5},"wasSuccessful":false}\nPERSISTENCE_RESTORE_FAILED'
    )
    permission = (
        'BackendError: POST failed with: {"errors":["Permission denied"],'
        '"error":{"code":5},"wasSuccessful":false}\nPERSISTENCE_RESTORE_FAILED'
    )
    assert classify(exact)["findings"][0]["classification"] == "BLOCKED_TEST_CONTRACT_DRIFT"
    assert classify(permission)["findings"][0]["classification"] == "EVIDENCE_GAP"
    assert classify("STAGE1 SHA mismatch")["findings"][0]["classification"] == (
        "BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH"
    )
    assert (
        classify("HTTP Error 403: Forbidden")["findings"][0]["confidence"] == "needs-reconciliation"
    )
    assert classify("Stopped thinking")["findings"][0]["classification"] == (
        "BLOCKED_CHAT_STREAM_INTERRUPTED_IDENTITY_DRIFT"
    )
    assert classify("ordinary successful run")["findings"] == []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, help="UTF-8 log file; stdin is used when omitted")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    parser.add_argument(
        "--self-test", action="store_true", help="run built-in deterministic checks"
    )
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        print("triage_incident self-test: PASS")
        return 0

    text = (
        args.log.read_text(encoding="utf-8-sig", errors="replace") if args.log else sys.stdin.read()
    )
    print(json.dumps(classify(text), indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
