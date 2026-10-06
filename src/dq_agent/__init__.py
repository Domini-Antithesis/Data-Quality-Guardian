"""Data Quality Auto-Fixing & Validation Agent.

Detect data-quality problems deterministically, let an LLM advise which fix to
apply and explain why, have a human approve each one, apply the fixes with
plain pandas, then re-validate and score the result.
"""

__version__ = "0.1.0"
