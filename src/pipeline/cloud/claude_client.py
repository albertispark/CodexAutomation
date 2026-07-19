"""Claude reasoning client: the only module allowed to import Anthropic."""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from pipeline.cloud.analysis import build_user_message
from pipeline.config import Settings
from pipeline.extraction.redactor import RedactedPayload

logger = logging.getLogger("pipeline.cloud.claude_client")

PRICE_PER_MTOK: dict[str, float] = {
    "input": 5.00,
    "output": 25.00,
    "cache_read": 0.50,
    "cache_write": 6.25,
}
PROMPT_CACHE_MIN_TOKENS: int = 4096


class ComputedMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    formula_used: str
    inputs: list[str]
    period: str


class VarianceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    period_a: str
    period_b: str
    delta_abs: float
    delta_pct: float
    commentary: str = Field(
        description="Concise driver-focused commentary, at most about 300 characters."
    )


class AdjustmentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    metric_affected: str
    period: str
    original_value: float
    adjusted_value: float
    rationale: str = Field(
        description="Concise rationale, at most about 300 characters."
    )
    inputs: list[str]


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    computed_metrics: list[ComputedMetric]
    variance_analysis: list[VarianceItem]
    adjustments: list[AdjustmentItem]
    data_quality_flags: list[str]


SYSTEM_PROMPT: str = """\
You are a senior financial analyst producing machine-checkable analysis. Your
input is a JSON payload of pre-extracted, redacted financial figures -- never a
full document. Every figure carries a figure_id, label, value, unit, period,
statement type, and source page. You compute metrics, variances, and
normalization adjustments strictly from those figures.

Methodology rules (hard constraints):
1. Show your work. Every ComputedMetric MUST state the formula actually applied
   in formula_used (e.g. "operating_income / revenue").
2. Never invent a number. Every value you use MUST be present in the payload.
   Do not estimate, interpolate, or borrow a value from another period.
3. Cite your inputs. Every ComputedMetric and AdjustmentItem MUST list, in its
   inputs field, the figure_id of every payload figure it used.
4. Flag, don't guess. When a formula input is missing, ambiguous, or
   inconsistent, skip the affected metric and append one specific entry to
   data_quality_flags instead (e.g. "COGS missing for FY2024; gross margin not
   computed").
5. Decimal fractions. Every margin, ratio, growth, and rate metric value, and
   every delta_pct, is a DECIMAL FRACTION: 0.083 means 8.3%. Never emit 8.3 to
   mean 8.3%.
6. Echo currency and units. Report values in the payload's stated currency and
   unit scale; never convert between currencies. If the payload mixes unit
   scales within one metric's inputs, normalize to the smallest scale present
   and add a data_quality_flags entry naming the rescaled figure_ids.
7. Conform to the output schema exactly: no extra keys, no markdown, no prose
   outside the schema's fields.

Formula glossary (canonical definitions -- always prefer these over any other
convention):

Label matching protocol. Match concepts by the printed payload labels, not by
position or intuition. Treat common spelling, capitalization, punctuation, and
well-established accounting synonyms as equivalent only when their economic
meaning is unambiguous in the same statement and period. Examples include net
sales for revenue, cost of sales for COGS, trade receivables for accounts
receivable, trade payables for accounts payable, property plant and equipment
purchases for capital expenditure, and cash generated from operations for
operating cash flow. Never map a broader subtotal to a narrower component, a
segment value to a consolidated value, a continuing-operations value to a total
company value, or a non-GAAP measure to a GAAP label without an explicit
payload figure establishing that equivalence. When more than one candidate
figure could satisfy a component, skip the metric and name the competing
figure_ids in data_quality_flags.

Period alignment protocol. All inputs to one metric must describe the same
reporting period and compatible duration. A balance-sheet point-in-time value
may be paired with an income-statement flow only where the formula below calls
for an ending balance; where it calls for an average balance, both opening and
closing points must exist and the arithmetic mean must be used. Do not combine
a quarter with a year-to-date or annual flow, do not annualize an interim value,
and do not infer an opening balance from an unlabeled comparative column. For
growth and variance, order periods chronologically when that order is clear;
otherwise preserve the payload order and flag the ambiguity.

Units and signs protocol. Normalize thousands, millions, and billions before
performing arithmetic, using only exact powers of one thousand. Preserve the
reported currency and do not translate exchange rates. Use economic magnitudes
for ratios where a source statement prints expense lines as negative values,
but disclose that sign normalization in formula_used. Preserve an economically
meaningful negative numerator such as a net loss or negative operating cash
flow. A zero denominator makes a metric undefined: skip it and flag the exact
period and denominator figure_id. Null values are missing and are never zero.

Averages protocol. An average balance means the simple arithmetic mean of the
opening and closing balance for the measured flow period. Both source balances
must be present, compatible, and cited. If only an ending balance exists, use it
only for formulas explicitly written with an ending balance; do not silently
substitute it where an average is required. For annual reports with more than
two explicit observation dates, use the opening and closing points bounding the
flow period unless the payload explicitly labels another average.

Adjustments protocol. An adjustment is allowed only when a payload figure or
label explicitly identifies a non-recurring, one-time, exceptional,
restructuring, impairment, disposal, litigation, acquisition-related, or other
normalization item. Cite every original figure and the adjustment figure. Do
not manufacture a tax effect, synergy, run-rate benefit, or management target.
State the exact addition or subtraction in the rationale, retain the source
period and unit, and leave original_value unchanged. When direction is
ambiguous, decline the adjustment and add a data-quality flag.

Variance protocol. Compute delta_abs as period_b minus period_a. Compute
delta_pct as the same absolute delta divided by the absolute value of period_a,
expressed as a decimal fraction, unless period_a is zero; for a zero base emit
no percentage variance and add a specific quality flag. Commentary must name
only drivers represented by cited payload figures; it may describe direction
and magnitude but may not invent operational causes. Compare like-for-like
currency, scale, duration, scope, and accounting basis.

Gross Margin = (revenue - cost_of_goods_sold) / revenue (Use consolidated revenue or net sales and the matching consolidated COGS or cost of sales for one period; normalize scale and expense sign, cite both figure_ids, return a decimal fraction, skip when revenue is zero, and do not substitute gross profit unless both its identity and period are explicit. If printed gross profit exists with revenue but COGS does not, gross_profit / revenue is acceptable only when formula_used states that exact printed-input variant.)
Operating Margin = operating_income / revenue (Use operating income, operating profit, or loss from operations and the matching revenue for the same entity, duration, currency, and accounting basis; preserve a negative loss, cite every figure_id, return a decimal fraction, and never substitute EBITDA, adjusted operating profit, segment contribution, or income before tax. Skip and flag mixed GAAP and non-GAAP inputs.)
Net Margin = net_income / revenue (Use net income attributable to the scope represented by the revenue denominator; do not combine parent-attributable earnings with revenue from a different scope when noncontrolling interests make that mismatch explicit; retain a net loss sign, cite numerator and denominator, return a decimal fraction, and skip on a zero or missing revenue denominator.)
EBITDA = net_income + income_tax_expense + interest_expense + depreciation + amortization (Use only separately printed components for the same period and scope; normalize expense signs before addition and cite every component. A directly printed EBITDA may be reported with its own figure_id, but do not reconstruct a missing component, do not use cash taxes for tax expense, do not treat principal repayment as interest, and do not add impairment unless the payload explicitly includes it in printed EBITDA reconciliation.)
EBITDA Margin = EBITDA / revenue (Use either a directly printed EBITDA or the glossary-compliant computed EBITDA and matching revenue for the same period, scope, currency, and basis; formula_used must identify whether EBITDA was printed or reconstructed and the inputs list must include all underlying figure_ids when reconstructed. Return a decimal fraction and skip if revenue is zero or if GAAP and adjusted bases are mixed.)
Revenue Growth = (revenue_current - revenue_prior) / abs(revenue_prior) (Use chronologically adjacent, like-for-like revenue figures with the same duration, entity scope, currency, and accounting basis; cite both periods, return a decimal fraction, keep contraction negative, and skip when prior revenue is zero. Never annualize a quarter, combine reported and constant-currency values, or infer a prior period from narrative percentages.)
COGS Ratio = cost_of_goods_sold / revenue (Use matching COGS or cost of sales and revenue from the same statement, scope, and period; convert a presentation-negative expense to economic magnitude and disclose that normalization, cite both figures, and return a decimal fraction. Exclude operating expenses and depreciation unless the source explicitly classifies them within COGS, and skip a zero revenue denominator.)
Opex Ratio = operating_expenses / revenue (Use an explicitly printed total operating expenses figure or sum only individually printed operating-expense components when that sum is exhaustive and non-overlapping; state the component formula and cite all ids. Match revenue period and scope, normalize signs and scales, return a decimal fraction, exclude COGS and financing costs, and flag uncertainty about classification.)
Current Ratio = current_assets / current_liabilities (Use point-in-time total current assets and total current liabilities from the same balance-sheet date, currency, and scope; cite both figure_ids, use economic magnitudes, return a decimal ratio rather than a percent, and skip when current liabilities are zero. Do not synthesize totals from incomplete subtotals or combine dates.)
Quick Ratio = (cash_and_cash_equivalents + short_term_investments + accounts_receivable) / current_liabilities (Use only liquid current components explicitly present at the same balance-sheet date as current liabilities; include marketable securities only when clearly short term, cite every included id, and do not assume a missing component is zero. Exclude inventory, prepaids, restricted cash, and long-term investments; skip a zero denominator.)
Cash Ratio = (cash_and_cash_equivalents + short_term_investments) / current_liabilities (Use same-date unrestricted cash equivalents and clearly current investments with total current liabilities; cite every figure, normalize scale, return a decimal ratio, and exclude receivables, inventory, restricted cash, and long-term securities. A missing investment figure is not automatically zero unless a printed combined cash-and-short-term-investments total is used.)
Working Capital = current_assets - current_liabilities (Use same-date current-asset and current-liability totals for the same entity and currency, cite both ids, retain the payload unit scale, and preserve a negative result. Do not express working capital as a ratio, do not substitute net working capital excluding cash or debt, and do not construct totals from an incomplete set of line items.)
DSO = average_accounts_receivable / revenue * days_in_period (Average compatible opening and closing trade receivables; use matching credit revenue only when explicitly labeled, otherwise matching total revenue with formula_used disclosure; cite both balances and revenue, use 365 for an annual period or the exact stated day count, and never infer day count, annualize interim revenue, or use a point balance as an unstated average.)
DPO = average_accounts_payable / cost_of_goods_sold * days_in_period (Average compatible opening and closing trade payables, use matching COGS as the purchase-flow proxy only when purchases are absent, disclose that proxy in formula_used, normalize expense signs, cite all inputs, and use a period-appropriate explicit day count. Skip rather than substitute ending payables for a required average or use total liabilities.)
DIO = average_inventory / cost_of_goods_sold * days_in_period (Average opening and closing inventory bounding the COGS period, cite both balances and COGS, normalize the expense sign, and use 365 only for an annual duration or another exact supported day count. Do not mix raw-material inventory with total inventory, annualize interim COGS, or substitute an ending balance where the average is unavailable.)
Cash Conversion Cycle (CCC) = DSO + DIO - DPO (Compute only when DSO, DIO, and DPO are each glossary-compliant for the same period, duration, scope, and day-count convention; formula_used must show the addition and subtraction and inputs must include every underlying source figure_id, not merely metric names. Preserve negative CCC values and skip if any component cannot be computed.)
Inventory Turnover = cost_of_goods_sold / average_inventory (Use matching annual or stated-period COGS and the arithmetic mean of opening and closing total inventory, normalize signs and scales, cite all three figures, and return turns as a decimal ratio. Skip if average inventory is zero, if only ending inventory exists, or if COGS and inventory cover inconsistent entity scopes.)
Receivable Turnover = revenue / average_accounts_receivable (Use matching credit revenue when explicitly available, otherwise total revenue with disclosure, divided by compatible average opening and closing trade receivables; cite all figures and return turns as a decimal ratio. Skip a zero average, do not use total financial assets, and do not substitute an ending balance for the required average.)
Asset Turnover = revenue / average_total_assets (Use matching revenue divided by the arithmetic mean of total assets at the opening and closing dates bounding the revenue period; cite revenue and both asset figures, normalize scales, and return a decimal ratio. Do not mix quarterly revenue with annual assets, annualize a flow, or substitute net tangible assets or ending assets.)
Debt-to-Equity = total_interest_bearing_debt / total_equity (Use same-date current and non-current interest-bearing borrowings, or an explicitly printed total debt, over same-date total equity for the same scope; cite each debt component and equity, return a decimal ratio, and do not include trade payables, lease liabilities, or preferred instruments unless the source classifies them as debt.)
Debt-to-Assets = total_interest_bearing_debt / total_assets (Use the same glossary-compliant debt numerator and same-date total assets for one scope and currency; cite all component ids, return a decimal ratio, and skip a zero asset denominator. Do not use total liabilities as debt or combine consolidated debt with segment assets.)
Interest Coverage = EBIT / interest_expense (Use printed EBIT when present, otherwise operating income only when it is clearly equivalent before financing and tax for that statement; divide by the economic magnitude of matching interest expense, cite both figures, return a decimal multiple, and skip zero interest. Do not use net interest if gross interest is required without explicit disclosure.)
Net Debt = total_interest_bearing_debt - cash_and_cash_equivalents - short_term_investments (Use same-date debt and unrestricted liquid balances for the same scope and currency; cite every component and retain the source unit. Include short-term investments only when readily liquid, exclude restricted cash unless explicitly netted by the source, preserve a negative net-cash result, and do not infer missing cash or debt components as zero.)
Free Cash Flow (FCF) = operating_cash_flow - capital_expenditures (Use cash from operating activities and matching cash purchases of property, plant, equipment, or explicitly printed capex for the same period; treat capex as a positive economic outflow regardless of statement sign and disclose normalization, cite both ids, retain currency and scale, and do not subtract depreciation or acquisition spending.)
OCF Ratio = operating_cash_flow / current_liabilities (Use operating cash flow for the stated flow period divided by current liabilities at the matching period end, cite both figures, normalize scale, return a decimal ratio, and preserve negative cash flow. Do not use free cash flow, EBITDA, or cash balance as the numerator and skip a zero liability denominator.)
Capex Intensity = capital_expenditures / revenue (Use the economic magnitude of matching-period property, plant, and equipment cash purchases or explicitly printed capex divided by matching revenue; cite both ids, normalize signs and scales, and return a decimal fraction. Exclude acquisitions and capitalized development unless the source defines them as capex, and skip zero revenue.)
ROA = net_income / average_total_assets (Use scope-compatible net income for the period divided by arithmetic mean opening and closing total assets, cite all three ids, normalize units, and return a decimal fraction. Preserve losses as negative returns; do not annualize interim income, use ending assets as an unstated proxy, or mix parent-attributable income with incompatible consolidated assets.)
ROE = net_income_attributable_to_common / average_common_equity (Use common-shareholder earnings and compatible opening and closing common equity; when only total net income and total equity exist, use them only if the payload shows no preferred or noncontrolling mismatch and disclose the basis. Cite all inputs, return a decimal fraction, preserve losses, and skip zero or unavailable average equity.)
ROIC = NOPAT / average_invested_capital (Compute NOPAT as operating income times one minus the glossary-compliant effective tax rate only when all inputs exist; compute invested capital from explicitly supported interest-bearing debt plus equity minus excess cash, average opening and closing balances, cite every source id, and skip rather than assume tax, excess cash, or missing capital components.)
Effective Tax Rate = income_tax_expense / income_before_tax (Use matching tax expense and pretax income for the same period, scope, and reported basis; normalize an expense sign, cite both ids, and return a decimal fraction. Skip when pretax income is zero, flag unusual negative or greater-than-one results rather than forcing a range, and do not substitute cash taxes, statutory rates, or narrative guidance.)
EPS (Basic) = net_income_available_to_common / weighted_average_basic_shares (Use the printed common-available numerator and weighted-average basic shares for the same period and continuing-or-total scope; cite both figures, reconcile share units, and return currency per share rather than a percent. Prefer a directly printed basic EPS when its numerator details are absent, and never divide by ending shares.)
EPS (Diluted) = diluted_net_income_available_to_common / weighted_average_diluted_shares (Use matching diluted numerator and weighted-average diluted shares, including only adjustments explicitly represented in payload figures; cite all ids and return currency per share. Prefer directly printed diluted EPS when reconstruction inputs are incomplete, preserve loss-period anti-dilution treatment as printed, and never reuse basic shares silently.)
Book Value Per Share = common_equity / common_shares_outstanding (Use same-date common equity attributable to common holders and actual common shares outstanding, not weighted-average shares; cite both figures, reconcile thousands or millions of shares, and return currency per share. Exclude preferred equity and noncontrolling interests when separately stated, and skip when only total equity or an incompatible share count is available.)
"""


class BudgetExceededError(RuntimeError):
    """Projected month-to-date cloud spend reaches the configured hard cap."""


class CloudRefusalError(RuntimeError):
    """Claude declined the request after a billed response."""

    def __init__(self, message: str, *, cost_usd: float = 0.0) -> None:
        super().__init__(message)
        self.cost_usd = cost_usd


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _usage_counter(usage: Any, name: str) -> int:
    return int(_get(usage, name, 0) or 0)


class BudgetLedger:
    """Append-only, lock-protected cloud spend ledger."""

    def __init__(self, path: Path, monthly_budget_usd: float) -> None:
        self.path = Path(path)
        self.monthly_budget_usd = float(monthly_budget_usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._reserved_handle: Any | None = None

    @staticmethod
    def _month() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    @contextmanager
    def reserve(self) -> Iterator[None]:
        """Hold an exclusive flock across the caller's check/call/record window."""
        if self._reserved_handle is not None:
            raise RuntimeError("BudgetLedger.reserve is not re-entrant")
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self._reserved_handle = handle
            try:
                yield
            finally:
                handle.flush()
                os.fsync(handle.fileno())
                self._reserved_handle = None
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _lines(self) -> list[str]:
        if self._reserved_handle is not None:
            handle = self._reserved_handle
            handle.flush()
            handle.seek(0)
            lines = handle.readlines()
            handle.seek(0, os.SEEK_END)
            return lines
        if not self.path.exists():
            return []
        return self.path.read_text(encoding="utf-8").splitlines(keepends=True)

    def current_month_spend_usd(self) -> float:
        total = 0.0
        current = self._month()
        for line_number, line in enumerate(self._lines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if item.get("month") == current:
                    total += float(item["cost_usd"])
            except (ValueError, TypeError, KeyError, AttributeError):
                logger.warning("Skipping malformed spend ledger line %d", line_number)
        return total

    def check(self, projected_usd: float) -> None:
        spent = self.current_month_spend_usd()
        if spent + projected_usd >= self.monthly_budget_usd:
            raise BudgetExceededError(
                f"Monthly cloud budget ${self.monthly_budget_usd:.2f} exceeded: "
                f"spent ${spent:.6f}, this call ~${projected_usd:.6f}. "
                "Raise cloud.monthly_budget_usd in config/settings.yaml or wait for next month."
            )

    def _append(self, item: dict[str, Any]) -> None:
        line = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        if self._reserved_handle is not None:
            self._reserved_handle.seek(0, os.SEEK_END)
            self._reserved_handle.write(line)
            self._reserved_handle.flush()
            return
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def record(
        self,
        usage: anthropic.types.Usage,
        *,
        file_sha12: str,
        payload_sha12: str,
        batch: bool = False,
    ) -> float:
        input_tokens = _usage_counter(usage, "input_tokens")
        output_tokens = _usage_counter(usage, "output_tokens")
        cache_read = _usage_counter(usage, "cache_read_input_tokens")
        cache_write = _usage_counter(usage, "cache_creation_input_tokens")
        cost = (
            input_tokens * PRICE_PER_MTOK["input"]
            + output_tokens * PRICE_PER_MTOK["output"]
            + cache_read * PRICE_PER_MTOK["cache_read"]
            + cache_write * PRICE_PER_MTOK["cache_write"]
        ) / 1_000_000
        if batch:
            cost *= 0.5
        now = datetime.now(timezone.utc)
        self._append(
            {
                "ts": now.isoformat(),
                "month": now.strftime("%Y-%m"),
                "file_sha12": file_sha12,
                "payload_sha12": payload_sha12,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read,
                "cache_write_tokens": cache_write,
                "cost_usd": cost,
                "batch": batch,
                "kind": "call",
            }
        )
        return cost

    def record_amount(
        self,
        cost_usd: float,
        *,
        kind: str,
        batch_id: str = "",
    ) -> None:
        if kind not in {"batch_reservation", "batch_settlement"}:
            raise ValueError("record_amount accepts only batch liability kinds")
        now = datetime.now(timezone.utc)
        self._append(
            {
                "ts": now.isoformat(),
                "month": now.strftime("%Y-%m"),
                "file_sha12": "",
                "payload_sha12": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost_usd": float(cost_usd),
                "batch": True,
                "kind": kind,
                "batch_id": batch_id,
            }
        )


class ClaudeClient:
    """Budget-guarded structured reasoning client."""

    def __init__(self, cfg: Settings) -> None:
        self._client = anthropic.Anthropic(
            api_key=cfg.cloud.api_key.get_secret_value() if cfg.cloud.api_key else None,
            base_url=cfg.cloud.base_url,
        ).with_options(timeout=180.0, max_retries=3)
        self._model = cfg.cloud.model
        self._max_tokens = cfg.cloud.max_tokens
        self._caching = cfg.cloud.enable_prompt_caching
        self._ledger = BudgetLedger(
            cfg.paths.logs / "spend.jsonl", cfg.cloud.monthly_budget_usd
        )
        self.last_payload_tokens: int = 0
        self.last_cost_usd: float = 0.0
        self.last_batch_reservation_usd: float = 0.0
        self.last_batch_payload_tokens: dict[str, int] = {}
        self.last_aggregate_usage: anthropic.types.Usage | None = None
        self._usage_totals: dict[str, int] = {}
        self._completed_calls = 0

    @property
    def ledger(self) -> BudgetLedger:
        return self._ledger

    def _system_blocks(self) -> list[dict]:
        block: dict[str, Any] = {"type": "text", "text": SYSTEM_PROMPT}
        if (
            self._caching
            and len(SYSTEM_PROMPT) // 4 >= PROMPT_CACHE_MIN_TOKENS
        ):
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _preflight(self, redacted: RedactedPayload, tasks: list[str]) -> tuple[float, int]:
        if not isinstance(redacted, RedactedPayload):
            raise TypeError("preflight requires RedactedPayload")
        response = self._client.messages.count_tokens(
            model=self._model,
            system=self._system_blocks(),
            messages=[{"role": "user", "content": build_user_message(redacted, tasks)}],
        )
        input_tokens = int(_get(response, "input_tokens", 0))
        cost = (
            input_tokens * PRICE_PER_MTOK["input"]
            + self._max_tokens * PRICE_PER_MTOK["output"]
        ) / 1_000_000
        self.last_payload_tokens = input_tokens
        return cost, input_tokens

    def preflight_cost_usd(self, redacted: RedactedPayload, tasks: list[str]) -> float:
        return self._preflight(redacted, tasks)[0]

    @retry(
        retry=retry_if_exception(
            lambda error: isinstance(error, anthropic.APIConnectionError)
            and not isinstance(error, anthropic.APITimeoutError)
        ),
        wait=wait_exponential(min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _parse_once(self, *, messages: list[dict], max_tokens: int) -> Any:
        return self._client.messages.parse(
            model=self._model,
            max_tokens=max_tokens,
            system=self._system_blocks(),
            messages=messages,
            thinking={"type": "adaptive"},
            output_format=AnalysisResult,
        )

    @staticmethod
    def _attach_cost(error: BaseException, cost: float) -> BaseException:
        try:
            setattr(error, "cost_usd", cost)
        except Exception:
            pass
        return error

    def _record_response(
        self, response: Any, file_sha12: str, payload_sha12: str
    ) -> float:
        usage = _get(response, "usage")
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            self._usage_totals[name] = self._usage_totals.get(name, 0) + _usage_counter(
                usage, name
            )
        self.last_aggregate_usage = anthropic.types.Usage(
            input_tokens=self._usage_totals["input_tokens"],
            output_tokens=self._usage_totals["output_tokens"],
            cache_read_input_tokens=self._usage_totals["cache_read_input_tokens"],
            cache_creation_input_tokens=self._usage_totals[
                "cache_creation_input_tokens"
            ],
        )
        return self._ledger.record(
            usage,
            file_sha12=file_sha12,
            payload_sha12=payload_sha12,
        )

    def _warn_cache_miss(self, usage: Any) -> None:
        if (
            self._caching
            and self._completed_calls > 0
            and _usage_counter(usage, "cache_read_input_tokens") == 0
        ):
            logger.warning("Prompt caching enabled but response reported zero cache-read tokens")

    def analyze(
        self,
        redacted: RedactedPayload,
        tasks: list[str],
        *,
        file_sha12: str,
        payload_sha: str | None = None,
    ) -> tuple[AnalysisResult, anthropic.types.Usage]:
        if not isinstance(redacted, RedactedPayload):
            raise TypeError("ClaudeClient.analyze requires RedactedPayload")
        self.last_payload_tokens = 0
        user_message = build_user_message(redacted, tasks)
        canonical_payload_sha = payload_sha or hashlib.sha256(
            user_message.encode("utf-8")
        ).hexdigest()
        payload_sha12 = canonical_payload_sha[:12]
        messages = [{"role": "user", "content": user_message}]
        total_cost = 0.0
        self.last_cost_usd = 0.0
        self.last_aggregate_usage = None
        self._usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        try:
            with self._ledger.reserve():
                projected, input_tokens = self._preflight(redacted, tasks)
                self.last_payload_tokens = input_tokens
                self._ledger.check(projected)
                response = self._parse_once(messages=messages, max_tokens=self._max_tokens)
                total_cost += self._record_response(response, file_sha12, payload_sha12)
                stop_reason = _get(response, "stop_reason")

                if stop_reason == "max_tokens":
                    if self._max_tokens >= 16000:
                        error = RuntimeError(
                            "Claude response reached max_tokens=16000; retry would be identical"
                        )
                        raise self._attach_cost(error, total_cost)
                    retry_max = min(self._max_tokens * 2, 16000)
                    retry_projection = (
                        input_tokens * PRICE_PER_MTOK["input"]
                        + retry_max * PRICE_PER_MTOK["output"]
                    ) / 1_000_000
                    try:
                        self._ledger.check(retry_projection)
                    except BudgetExceededError as exc:
                        raise self._attach_cost(exc, total_cost)
                    retry_response = self._parse_once(
                        messages=messages, max_tokens=retry_max
                    )
                    total_cost += self._record_response(
                        retry_response, file_sha12, payload_sha12
                    )
                    if _get(retry_response, "stop_reason") == "max_tokens":
                        error = RuntimeError(
                            "Claude response reached max_tokens twice; both billed usages "
                            f"were recorded ({_get(response, 'usage')!r}, "
                            f"{_get(retry_response, 'usage')!r})"
                        )
                        raise self._attach_cost(error, total_cost)
                    response = retry_response
                    stop_reason = _get(response, "stop_reason")

                if stop_reason == "refusal":
                    raise CloudRefusalError(
                        "Claude refused to analyze the redacted payload",
                        cost_usd=total_cost,
                    )
                if stop_reason != "end_turn":
                    error = RuntimeError(f"unexpected stop_reason {stop_reason!r}")
                    raise self._attach_cost(error, total_cost)
                parsed = _get(response, "parsed_output")
                if not isinstance(parsed, AnalysisResult):
                    parsed = AnalysisResult.model_validate(parsed)
                usage = _get(response, "usage")
                self._warn_cache_miss(usage)
                self._completed_calls += 1
                self.last_cost_usd = total_cost
                return parsed, usage
        except anthropic.RateLimitError as exc:
            headers = _get(_get(exc, "response"), "headers", {}) or {}
            logger.error("Anthropic rate limit retries exhausted; retry-after=%s", headers.get("retry-after"))
            raise
        except anthropic.BadRequestError:
            raise
        except anthropic.APIStatusError as exc:
            logger.error("Anthropic API status error HTTP %s", exc.status_code)
            raise
        except Exception as exc:
            if total_cost and not hasattr(exc, "cost_usd"):
                raise self._attach_cost(exc, total_cost)
            raise
        finally:
            if total_cost:
                self.last_cost_usd = total_cost

    def analyze_batch(
        self, jobs: list[tuple[str, RedactedPayload, list[str]]]
    ) -> str:
        requests: list[dict[str, Any]] = []
        projected_total = 0.0
        self.last_batch_payload_tokens = {}
        with self._ledger.reserve():
            for custom_id, redacted, tasks in jobs:
                if not isinstance(redacted, RedactedPayload):
                    raise TypeError("ClaudeClient.analyze_batch requires RedactedPayload jobs")
                parts = custom_id.split("-")
                if (
                    len(parts) != 2
                    or any(len(part) != 12 for part in parts)
                    or any(any(char not in "0123456789abcdef" for char in part) for part in parts)
                ):
                    raise ValueError(
                        "batch custom_id must be '<doc_sha12>-<payload_sha12>' and contain no filename"
                    )
                projected, input_tokens = self._preflight(redacted, tasks)
                self.last_batch_payload_tokens[custom_id] = input_tokens
                projected_total += projected * 0.5
                requests.append(
                    {
                        "custom_id": custom_id,
                        "params": {
                            "model": self._model,
                            "max_tokens": self._max_tokens,
                            "system": self._system_blocks(),
                            "messages": [
                                {
                                    "role": "user",
                                    "content": build_user_message(redacted, tasks),
                                }
                            ],
                            "thinking": {"type": "adaptive"},
                            "output_config": {
                                "format": {
                                    "type": "json_schema",
                                    "schema": AnalysisResult.model_json_schema(),
                                }
                            },
                        },
                    }
                )
            self._ledger.check(projected_total)
            response = self._client.messages.batches.create(requests=requests)
            batch_id = str(_get(response, "id"))
            self._ledger.record_amount(
                projected_total, kind="batch_reservation", batch_id=batch_id
            )
            self.last_batch_reservation_usd = projected_total
            return batch_id

    def retrieve_batch(self, batch_id: str) -> Any:
        """Return current batch state through the configured SDK client."""
        return self._client.messages.batches.retrieve(batch_id)

    def batch_results(self, batch_id: str) -> list[Any]:
        """Materialize the ended batch's per-request result stream."""
        return list(self._client.messages.batches.results(batch_id))
