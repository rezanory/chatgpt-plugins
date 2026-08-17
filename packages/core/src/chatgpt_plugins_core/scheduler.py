from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from .models import AccountDescriptor, JobRequest


class NoEligibleAccount(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AccountLease:
    account_id: str
    job_id: str


class AccountScheduler:
    """Thread-safe, quota-respecting scheduler for authorized provider accounts.

    This is a provider-neutral in-process planning primitive. The active V0.1 GitHub-Issue
    transport uses explicit unique account IDs per batch; transactional cross-Issue leases are
    intentionally deferred to V0.2.
    """

    def __init__(self, accounts: Iterable[AccountDescriptor]) -> None:
        self._accounts = {a.account_id: a for a in accounts}
        self._active: Counter[str] = Counter()
        self._jobs: dict[str, str] = {}
        self._lock = threading.RLock()

    def active_count(self, account_id: str) -> int:
        with self._lock:
            return self._active[account_id]

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {account_id: self._active[account_id] for account_id in self._accounts}

    def restore_active(self, account_id: str, job_id: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                return
            account = self._accounts.get(account_id)
            if account is None:
                return
            self._active[account_id] += 1
            self._jobs[job_id] = account_id

    def acquire(self, job: JobRequest) -> AccountLease:
        with self._lock:
            if job.job_id in self._jobs:
                return AccountLease(self._jobs[job.job_id], job.job_id)
            candidates = [a for a in self._accounts.values() if self._eligible(a, job)]
            preferred_rank = {a: i for i, a in enumerate(job.preferred_account_ids)}
            candidates.sort(
                key=lambda a: (
                    0 if a.account_id in preferred_rank else 1,
                    preferred_rank.get(a.account_id, 10_000),
                    self._score(a),
                    a.account_id,
                )
            )
            if not candidates:
                raise NoEligibleAccount(f"no eligible account for job {job.job_id}")
            chosen = candidates[0]
            self._active[chosen.account_id] += 1
            self._jobs[job.job_id] = chosen.account_id
            return AccountLease(chosen.account_id, job.job_id)

    def release(self, job_id: str) -> bool:
        with self._lock:
            account_id = self._jobs.pop(job_id, None)
            if account_id is None:
                return False
            if self._active[account_id] > 0:
                self._active[account_id] -= 1
            return True

    def account(self, account_id: str) -> AccountDescriptor:
        return self._accounts[account_id]

    def accounts(self) -> list[AccountDescriptor]:
        return list(self._accounts.values())

    def _eligible(self, account: AccountDescriptor, job: JobRequest) -> bool:
        return (
            account.enabled
            and job.required_capabilities.issubset(account.capabilities)
            and job.required_labels.issubset(account.labels)
            and self._active[account.account_id] < account.max_parallel
        )

    def _score(self, account: AccountDescriptor) -> float:
        # Lower score wins. Weight lets an operator prefer a larger/healthier account without
        # pretending its provider quota is larger than it really is.
        return (self._active[account.account_id] / account.max_parallel) / account.weight
