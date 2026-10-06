# Evaluation

The programme guidance says to evaluate before deploying. For a data-quality tool the question is not "how accurate is the model" but "does it find what is actually wrong, and do the fixes actually fix it". This file records what was measured.

## Ground truth: the bundled samples

`scripts/make_samples.py` generates two datasets with a fixed seed, so the committed files are reproducible and every flaw in them was planted deliberately.

**`messy_sales_data.csv`** — 249 rows, 13 columns, with seven kinds of planted flaw:

| Planted | How |
|---|---|
| Three date formats | `2026-01-05`, `05/01/2026`, `5 January 2026` |
| Inconsistent casing | City and SalesRep randomly upper-cased or lower-cased |
| Stray whitespace | City values padded with spaces |
| A mixed-type column | Revenue mostly numeric with `N/A`, `pending`, `TBD`, `-` mixed in |
| Missing values | DiscountPct about 20% blank, Notes about 60% blank |
| Invalid emails | About 7% malformed |
| A constant column | Currency is `INR` in every row |
| Outliers | Four UnitPrice values multiplied by 100, a plausible extra-zero typo |
| Duplicate rows | Nine exact copies |

**`clean_orders.csv`** — 60 rows with nothing wrong, so the app can be checked for false positives.

## Detector results

```
$ dq-agent scan src/dq_agent/resources/samples/messy_sales_data.csv

messy_sales_data.csv: quality score 71.0/100 (Fair)
  completeness 62.9  uniqueness 98.4  validity 92.1  consistency 74.0

  11 issues:
   [critical] 153 blank values in 'Notes'                      (61.45%)
   [high    ] 'OrderDate' uses 3 different date formats        (38.55%)
   [high    ] 51 blank values in 'DiscountPct'                 (20.48%)
   [high    ] 'Revenue' mixes numbers and text                 (4.82%)
   [high    ] 11 values in 'CustomerEmail' are not a valid email address (4.42%)
   [medium  ] 'City' has 6 values written in different cases   (100.0%)
   [medium  ] 'SalesRep' has 5 values written in different cases (100.0%)
   [medium  ] 9 duplicate rows                                 (3.61%)
   [medium  ] 4 extreme values in 'UnitPrice'                  (1.61%)
   [low     ] 'Currency' has the same value in every row       (100.0%)
   [low     ] 20 values in 'City' have stray spaces            (8.03%)
```

**Every planted flaw was found, and nothing was invented.** Nine detectors produced eleven issues because casing and whitespace both appear in more than one column.

```
$ dq-agent scan src/dq_agent/resources/samples/clean_orders.csv

clean_orders.csv: quality score 100.0/100 (Excellent)
  No issues found.
```

**Zero false positives on the clean dataset.** This is the check worth trusting least by default and worth running most often: a detector that flags everything is useless, and the clean sample is the guard against that.

## End-to-end results

Both runs approve the recommended fix for every issue and then re-run all nine detectors on the result.

| Advisor | Score before | Score after | Fixes applied | Rows | Columns | Time |
|---|---|---|---|---|---|---|
| Groq `openai/gpt-oss-120b` | 71.0 | **98.3** | 11 | 249 → 240 | 13 → 12 | about 7 s |
| Built-in rules (Demo mode) | 71.0 | **86.8** | 8 | 249 → 240 | 13 → 13 | under 1 s |

The live model scores higher because it acts on three issues the conservative rules leave alone: it clips the outliers, drops the constant `Currency` column, and fills the mostly-empty `Notes` column.

**That last choice is worth looking at closely.** The rules refuse to fill a column that is 61% empty, on the grounds that filling it would invent most of the data. The model filled all 153 blanks with the most common existing value, `Priority delivery`, which is almost certainly wrong for a free-text notes field.

This is not a bug in the model or the rules. It is precisely why the human approval step exists, and it is the single most useful thing the evaluation surfaced: **a higher quality score is not the same as better data.** The UI shows the reasoning behind every recommendation so the user can reject exactly this kind of choice, and the "Still outstanding" section afterwards reports honestly on what was left alone.

## Automated tests

`pytest tests/ -q`: **92 tests, all offline, about 7 seconds.** No network, no API key.

| Area | File | What is covered |
|---|---|---|
| Detectors | `test_detectors.py` | Every planted flaw found; zero issues on clean data; determinism; exact counts and severities; thresholds actually change behaviour; outliers need rows and spread; a flat column produces none; placeholder text counted as missing and called out; date shapes ignore day and month name; single-column and semicolon and tab files load; value classifiers; profiling counts |
| Fixes | `test_fixes.py` | Each strategy does what it claims and nothing more; trimming does not remove rows; blanking keeps the rest of the row; filling with an average converts the column; fill-constant needs a value; an average is refused on a text column; an illegal strategy is rejected; `none` is recorded; every strategy in every menu runs or refuses cleanly; ordering puts trimming before deduplication; rejected decisions do nothing; a fix on a missing column is skipped not crashed; the score genuinely rises |
| Pipeline | `test_pipeline.py` | Both graphs' nodes; the analyse phase produces no modified data; the apply phase re-validates rather than assuming; approving nothing changes nothing; the rule advisor covers every issue type with real reasoning; it refuses to invent data for a mostly-empty column; LLM advice parsed with the menu in the prompt; an out-of-menu strategy is rejected and repaired; two illegal answers raise; missing ids rejected; fenced and wrapped JSON tolerated; auth and model errors are not recoverable; token accounting; provider outage falls back with a warning; a rejected key stops the run; the advised-issue limit is respected |
| API | `test_api.py` | Health, index, static files; strategy help covers every strategy; settings round-trip with secret masking; **analysis stops for approval and produces no file**; approving improves the score and records every actor in the audit trail; rejecting everything changes nothing; an illegal fix is refused; unknown issue and unknown strategy rejected; preview before and after; a clean file reports nothing; bad uploads explained; history and deletion; the upload is kept byte-for-byte; no import-time side effects; the audit report escapes table characters |
| Config | `test_config.py` | Every setting documented in `.env.example` (fails on drift); every provider has a default model; Demo mode never claims a real model; precedence; a bad env value falls back; secrets never leak; `data_dir` is never persisted; tracing applied and cleared; bounds enforced; every issue type has a menu that includes `none`; every strategy has plain-language help; severity weights increase |

### Browser testing

The UI was driven end to end in real Microsoft Edge with **38 assertions**, all passing: every tab, the clean and messy samples, the upload path, 11 issue cards with severity badges, evidence and recommendations, the four quality dimensions, approve-all / approve-none / only-non-destructive, the destructive warning, changing a strategy and seeing the help text update, revealing the fill-value box and confirming the typed value reached the cleaned data, applying with confirmation, the before-and-after preview, both downloads, history, reopening a past run, the connection test, provider switching, an unsupported file, phone width and dark mode.

### Live verification

- Groq classification and recommendation on the full 249-row sample, with reasoning, in about 7 seconds.
- LangSmith tracing recording each workflow node as its own span.
- The wrong-key path: the run fails with a readable message instead of quietly producing rule-based advice.
- The unavailable-model path, found for real when Groq retired `llama-3.3-70b-versatile`.

## Known limitations

- **The score is a comparison tool, not an absolute measure.** Its weights are chosen, not derived. It is meaningful between the two passes of one dataset; comparing the scores of two unrelated datasets means less.
- **A higher score can mean worse data.** Filling a mostly-empty column raises completeness while inventing values. The app shows the reasoning and leaves the decision to a person; it does not, and should not, optimise the number.
- **Outlier detection is inter-quartile range only.** It assumes a roughly unimodal distribution. A genuinely bimodal column will have its smaller mode flagged.
- **Invalid-format checks are name-driven.** Only columns named like an email or phone are checked, because a rule the user cannot predict is worse than no rule. A column called `contact_1` is not checked.
- **Duplicate detection is exact-match only.** `"Jon Smith"` and `"John Smith"` are two different rows. Fuzzy entity resolution is a different project (capstone #41).
- **Ambiguous dates fall back to month-first.** When no value in the column has a first component above 12 there is no evidence either way. The applied-fix detail states which reading was used.
- **Whole-file, in-memory.** The 200,000-row cap is real. Anything larger belongs in a chunked or Spark-based pipeline.
- **No cross-column or business rules.** "Ship date must be after order date" and "revenue equals quantity times price" are not checked. Both are natural next additions and neither needs an LLM.
- **The LLM sees summaries, not your data.** It gets each issue's description and up to five example values. It cannot reason about patterns it was not shown, which bounds both the privacy exposure and the insight.
