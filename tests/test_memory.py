from decimal import Decimal

import pytest

from max_agent.memory import MemoryStore, Update, budget_snapshot


def test_intake_survives_restart_and_is_private(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    result = store.patch(
        "alice",
        [
            Update(field="name", value="Alice"),
            Update(field="currency", value="USD"),
            Update(field="income_monthly", value="4000"),
        ],
        source_message_id="turn-1",
    )
    assert result["income_monthly"]["value"] == "4000.00"
    reopened = MemoryStore(path)
    assert reopened.load("alice")["name"]["value"] == "Alice"
    assert reopened.load("bob") == {}


def test_invalid_batch_does_not_partially_save(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    with pytest.raises(ValueError):
        store.patch(
            "alice",
            [Update(field="name", value="Alice"), Update(field="income_monthly", value="-10")],
            "turn-1",
        )
    assert store.load("alice") == {}


def test_correction_omission_zero_unknown_and_skipped(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.patch(
        "alice",
        [Update(field="name", value="Alice"), Update(field="debt_payments_monthly", value="0")],
        "turn-1",
    )
    result = store.patch(
        "alice",
        [
            Update(field="name", value="Alicia"),
            Update(field="savings_balance", status="unknown"),
            Update(field="household", status="skipped"),
        ],
        "turn-2",
    )
    assert result["name"]["value"] == "Alicia"
    assert result["debt_payments_monthly"]["value"] == "0.00"
    assert result["savings_balance"]["status"] == "unknown"
    assert result["household"]["status"] == "skipped"
    result = store.patch("alice", [Update(field="name", status="cleared")], "turn-3")
    assert result["name"]["value"] is None
    assert result["debt_payments_monthly"]["value"] == "0.00"


def test_budget_uses_exact_monthly_amounts_and_flags_missing(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    assert budget_snapshot({})["status"] == "incomplete"
    facts = store.patch(
        "alice",
        [
            Update(field=k, value=v)
            for k, v in {
                "currency": "USD",
                "income_monthly": "4000.10",
                "fixed_expenses_monthly": "1800",
                "variable_expenses_monthly": "900.05",
                "debt_payments_monthly": "200",
                "savings_target_monthly": "500",
            }.items()
        ],
        "turn-1",
    )
    result = budget_snapshot(facts)
    assert Decimal(result["remaining_after_savings"]) == Decimal("600.05")
    facts = store.patch("alice", [Update(field="income_monthly", value="1000")], "turn-2")
    assert budget_snapshot(facts)["remaining_after_savings"] == "-2400.05"


def test_no_message_cannot_write(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    with pytest.raises(ValueError):
        store.patch("alice", [Update(field="name", value="Alice")], "")
    assert store.load("alice") == {}


def test_currency_cannot_relabel_existing_money(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.patch(
        "alice",
        [Update(field="currency", value="USD"), Update(field="income_monthly", value="4000")],
        "turn-1",
    )
    for change in [Update(field="currency", value="EUR"), Update(field="currency", status="cleared")]:
        with pytest.raises(ValueError):
            store.patch("alice", [change], "turn-2")
    assert store.load("alice")["currency"]["value"] == "USD"


def test_restatement_is_noop_and_other_session_fields_survive(tmp_path):
    path = tmp_path / "memory.db"
    first, second = MemoryStore(path), MemoryStore(path)
    original = first.patch("alice", [Update(field="name", value="Alice")], "turn-1")
    second.patch("alice", [Update(field="household", value="Lives alone")], "turn-2")
    updated = first.patch("alice", [Update(field="name", value="Alice")], "turn-3")
    assert updated["name"] == original["name"]
    assert updated["household"]["value"] == "Lives alone"
