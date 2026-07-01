# Automated Data Cleaning & EDA Crew

A [CrewAI](https://crewai.com) multi-agent system that profiles, cleans, and writes
an executive summary for a multi-sheet Excel dataset — so you don't have to write
the first round of exploratory pandas by hand.

Built and tested against `AAZ_LifeScience_Data 1.xlsx` (an Al Ain Zoo
life-sciences export: Animal Master, Lab Master, Necropsy, Medical Case, Case
Journal). Powered by **Google Gemini 3.1 Flash Lite** via LiteLLM.

## The crew

| Agent | Role | Tools | Output |
|---|---|---|---|
| 1. **Data Profiler** | Senior Data Profiler | `Profile dataset` | A per-sheet data-quality report: dtypes, missing %, uniqueness, IQR outliers, duplicates, type red-flags |
| 2. **Code/Viz Developer** | Python Visualization Developer | `Clean dataset`, `Visualize dataset` | Cleaned CSVs + histograms / missingness / correlation plots, described in words |
| 3. **Insights Summarizer** | Lead Data Analyst | — | A one-page executive summary of trends, data-quality issues, and next steps |

Agents run **sequentially**, each passing its findings to the next.

### Design choice: deterministic tools, not LLM-generated code

The agents call **deterministic Python tools** (real pandas/matplotlib in
[`tools/eda_tools.py`](src/data_cleaning_eda_crew/tools/eda_tools.py)) rather than
having the model write and execute pandas live. This means:

- Plots and profiles **always** generate — no Docker, no flaky code execution.
- The cheap/fast Gemini model spends its budget on *reasoning and writing*, which
  it's good at, not on *generating correct code*, which small models fumble.
- Because the LLM can't see images, each visualization tool also returns a **text
  description** (distribution skew, strongest correlations) so the summarizer can
  reason from text alone.

## Setup

```bash
# from the project root
uv sync                      # install dependencies into .venv
cp .env.example .env         # then edit .env with your key + data path
```

`.env` keys:

```ini
MODEL=gemini/gemini-3.1-flash-lite
GEMINI_API_KEY=your-key-here
DATA_FILE=C:\path\to\your\dataset.xlsx
```

## Run

```bash
uv run run-crew
```

## Outputs (written to `./output/`)

```
output/
├── reports/executive_summary.md   # the final one-page briefing
├── figures/                       # missingness, histograms, correlation PNGs per sheet
└── cleaned/                       # cleaned CSV per numeric sheet
```

## Pointing it at a different dataset

The crew is data-agnostic for Excel workbooks. To analyze your own file:

1. Set `DATA_FILE` in `.env`.
2. In [`tools/eda_tools.py`](src/data_cleaning_eda_crew/tools/eda_tools.py), edit
   `NUMERIC_SHEETS` (sheets that get plots + cleaning) and `TEXT_SHEETS`
   (free-text sheets that are profiled only).

## Notes

- **Cleaning is conservative and reversible**: it drops fully-empty columns and
  exact duplicate rows, trims whitespace, and converts `YYYYMMDD` integer
  date-keys to real dates. It deliberately does **not** impute missing values —
  it reports them for a human to decide.
- ID / surrogate-key / date-key columns are excluded from histograms and
  correlation matrices so the plots stay meaningful.
