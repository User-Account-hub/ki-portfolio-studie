# Security Review — Follow-up Actions

External code-security review, 11–12 Sep 2026 (commit `cea8255`). This file
tracks findings that are **not** fixed by a code change — operational actions
and recommendations. The code-fixable findings are separate PRs.

## Automated scan results (clean)
- **Secret scan** (gitleaks, full 28-commit history): 0 leaks. `.env` is git-ignored and untracked; only `.env.example` (placeholders) is committed; no key material in the committed `db/portfolio.db` or `reports/`. **Added 2026-09-18:** a `detect-secrets` pre-commit hook (`.pre-commit-config.yaml`, `.secrets.baseline`) now blocks new secret commits going forward, enforced both locally (`pre-commit install`) and in CI (`.github/workflows/pre-commit.yml`) so a skipped local hook doesn't leave a gap.
- **SAST** (semgrep, 290 rules): 0 findings.
- **SQL**: fully parameterised — no injection.
- **Dependencies** (OSV, PyPI): 0 findings at CVSS ≥ 7.0. See INFO-7.

---

## HIGH — Unresolved unauthorised trading activity (operational)
`INCIDENT_2026-09-08.md` documents 16 buy orders (~USD 599k notional) on the
Alpaca account that match **no** pipeline run, GitHub Action, or known session
(sub-millisecond fills, ~9-dp fractional quantities — inconsistent with the
pipeline). Root cause was never established, and the investigation itself flags
the **Alpaca activity log (API-key context / source IP)** as the only source
never examined. The same keys remain live in GitHub Actions Secrets and in daily
use. This is an unresolved credential-integrity signal, independent of the
paper-money caveat.

**Actions**
- [ ] Rotate `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` and `ANTHROPIC_API_KEY`.
- [ ] Pull the Alpaca **Activity / audit log** for 2026-09-08 ~08:03 UTC, filtered by API-key and **source IP** (the decisive, unexamined evidence).
- [ ] Review the GitHub Actions run history and secret-access log for that window.
- [ ] Enable branch protection + required review on `master` (the pipeline pushes with `contents: write`).
- [ ] Confirm this key/secret-handling pattern was never reused with a real-money Alpaca account.

## LOW — Full decision ledger + raw prompts/responses committed to git
The workflow commits `db/portfolio.db` (with full `prompt` and `raw_response`
text) and every report back to the repo. The repo is currently **private**, so
exposure is limited and no credentials are stored — but the entire strategy,
prompt engineering and portfolio state live in git history and would be fully
disclosed if the repo is ever made public.

**Recommendation:** keep it private; treat "make public" as requiring a history
scrub first, or move the ledger/reports to a workflow artifact / dedicated
private data branch rather than the code repo.

## ~~INFO — Dependency pinning / supply chain~~ (addressed 2026-09-18)
~~All 9 dependencies use unpinned `>=` constraints with no lockfile, so CI is not
reproducible and a future release resolves in unseen.~~ Today's OSV scan found
nothing ≥ CVSS 7.0 (the Low/Med hits — python-dotenv `set_key`, pytest tmpdir,
anthropic memory-tool, pydantic ReDoS — are outside the permitted range or
unreachable by how the code uses the libs), but the posture was the exact
condition that lets a medium CVE slip in unnoticed.

**Done:** all 9 direct dependencies in `requirements.txt` are now pinned to
exact, test-verified versions (`==`, not `>=`); Dependabot
(`.github/dependabot.yml`) opens weekly update PRs for both the `pip` and
`github-actions` ecosystems. **Still open (manual, repo Settings):** enable
"Dependabot alerts" + "Dependabot security updates" under *Settings → Code
security* — not settable via a committed file. **Not done:** a full
hash-locked transitive-dependency lockfile (pip-tools / uv) - a separate,
larger step than pinning the direct dependencies; splitting dependency
install from the `contents: write` commit step in CI remains open too.

## INFO — No retry/backoff or failure alerting
A failed Anthropic/Alpaca call ends the run with exit 1, visible only in the
Actions log; there is no notification on failures or on forced short-stop-loss
closures. Robustness/observability, not a security defect (already listed as a
known limitation in the README).

---
*Prepared as part of an external SECURIX code-security review. Findings are
documented, not exploited; no changes were made to production systems.*
