"""Atomic, per-user facts. Omitted fields are preserved; clearing is explicit."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

Field = Literal[
    "name",
    "currency",
    "household",
    "work_and_pay_schedule",
    "goals",
    "income_monthly",
    "fixed_expenses_monthly",
    "variable_expenses_monthly",
    "debt_payments_monthly",
    "debt_balance",
    "savings_balance",
    "savings_target_monthly",
]
Status = Literal["known", "unknown", "skipped", "cleared"]
MONEY = {
    "income_monthly",
    "fixed_expenses_monthly",
    "variable_expenses_monthly",
    "debt_payments_monthly",
    "debt_balance",
    "savings_balance",
    "savings_target_monthly",
}
CORE = (
    "currency",
    "income_monthly",
    "fixed_expenses_monthly",
    "variable_expenses_monthly",
    "debt_payments_monthly",
    "savings_target_monthly",
)


class Update(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Field
    status: Status = "known"
    value: str | None = None

    def normalized_value(self) -> str | None:
        if self.status != "known":
            if self.value is not None:
                raise ValueError("Non-known facts must omit the value.")
            return None
        if not self.value or not self.value.strip() or len(self.value) > 600:
            raise ValueError("Known facts need a nonempty value of at most 600 characters.")
        value = self.value.strip()
        if self.field in MONEY:
            try:
                amount = Decimal(value)
                if not amount.is_finite() or amount < 0 or amount > 1_000_000_000:
                    raise ValueError("Amounts must be finite, nonnegative and at most one billion.")
                if amount != amount.quantize(Decimal("0.01")):
                    raise ValueError("Amounts can have at most two decimal places.")
                return str(amount.quantize(Decimal("0.01")))
            except InvalidOperation as exc:
                raise ValueError("Use a numeric amount without currency signs or separators.") from exc
        if self.field == "currency":
            if len(value) != 3 or not value.isascii() or not value.isalpha():
                raise ValueError("Use a three-letter currency code such as USD.")
            return value.upper()
        return value


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(self._connect()) as conn, conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS profiles (
                user_id TEXT PRIMARY KEY, facts TEXT NOT NULL
            )""")
        self.path.chmod(0o600)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def load(self, user_id: str) -> dict:
        if not user_id:
            raise ValueError("A trusted user identity is required.")
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT facts FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def patch(self, user_id: str, updates: list[Update], source_message_id: str) -> dict:
        if not user_id or not source_message_id:
            raise ValueError("Trusted identity and a real user message are required.")
        if not updates or len(updates) > 12 or len({u.field for u in updates}) != len(updates):
            raise ValueError("Provide one to twelve updates with distinct fields.")
        normalized = [(u, u.normalized_value()) for u in updates]
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT facts FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
            facts = json.loads(row[0]) if row else {}
            # Currency changes need the amounts restated or cleared; never relabel old money.
            next_currency = next((v for u, v in normalized if u.field == "currency"), None)
            old_currency = facts.get("currency", {}).get("value")
            if any(u.field == "currency" for u in updates) and old_currency and next_currency != old_currency:
                existing_money = {k for k in MONEY if facts.get(k, {}).get("status") == "known"}
                if not existing_money.issubset({u.field for u in updates}):
                    raise ValueError("Changing currency requires restating or clearing all saved amounts.")
            for update, value in normalized:
                old = facts.get(update.field, {})
                if old.get("status") == update.status and old.get("value") == value:
                    continue
                facts[update.field] = {
                    "status": update.status,
                    "value": value,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "source_message_id": source_message_id,
                }
            conn.execute(
                "INSERT INTO profiles VALUES (?, ?) ON CONFLICT(user_id) "
                "DO UPDATE SET facts = excluded.facts",
                (user_id, json.dumps(facts)),
            )
        return facts


def budget_snapshot(facts: dict) -> dict:
    missing = [key for key in CORE if facts.get(key, {}).get("status") != "known"]
    if missing:
        return {"status": "incomplete", "missing": missing}
    values = {key: Decimal(facts[key]["value"]) for key in CORE if key != "currency"}
    expenses = sum(
        values[key]
        for key in ("fixed_expenses_monthly", "variable_expenses_monthly", "debt_payments_monthly")
    )
    surplus = values["income_monthly"] - expenses
    return {
        "status": "complete",
        "currency": facts["currency"]["value"],
        "period": "monthly",
        "income": str(values["income_monthly"]),
        "expenses_including_debt": str(expenses),
        "available_before_savings": str(surplus),
        "savings_target": str(values["savings_target_monthly"]),
        "remaining_after_savings": str(surplus - values["savings_target_monthly"]),
        "basis": "user-reported estimates, not verified account balances",
    }
