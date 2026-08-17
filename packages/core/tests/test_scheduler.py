from chatgpt_plugins_core import (
    AccountDescriptor,
    AccountScheduler,
    JobRequest,
    NoEligibleAccount,
    SourceRef,
)


def account(account_id: str, caps: set[str], max_parallel: int = 1, weight: float = 1.0):
    return AccountDescriptor(
        account_id=account_id,
        provider="kaggle",
        secret_scope=account_id,
        capabilities=frozenset(caps),
        max_parallel=max_parallel,
        weight=weight,
    )


def job(job_id: str, caps: set[str]):
    return JobRequest(job_id, SourceRef("owner/repo", "abc"), "tests", frozenset(caps))


def test_scheduler_balances_accounts_and_releases_by_job():
    scheduler = AccountScheduler([account("a", {"cpu"}), account("b", {"cpu"})])
    first = scheduler.acquire(job("j1", {"cpu"}))
    second = scheduler.acquire(job("j2", {"cpu"}))
    assert {first.account_id, second.account_id} == {"a", "b"}
    assert scheduler.release("j1") is True
    assert scheduler.release("j1") is False


def test_scheduler_respects_capabilities_and_parallel_limit():
    scheduler = AccountScheduler([account("cpu", {"cpu"}), account("gpu", {"cpu", "gpu"})])
    assert scheduler.acquire(job("j1", {"gpu"})).account_id == "gpu"
    try:
        scheduler.acquire(job("j2", {"gpu"}))
    except NoEligibleAccount:
        pass
    else:
        raise AssertionError("expected NoEligibleAccount")


def test_restore_active_rehydrates_load():
    scheduler = AccountScheduler([account("a", {"cpu"}, max_parallel=2)])
    scheduler.restore_active("a", "existing")
    assert scheduler.active_count("a") == 1
    scheduler.acquire(job("new", {"cpu"}))
    assert scheduler.active_count("a") == 2
