"""Entry point: load env, run the crew, report where outputs landed."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# crewai's verbose logger emits emoji; the default Windows console (cp1252)
# can't encode them. Force UTF-8 so logging never errors.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv

load_dotenv()  # load .env before the crew/tools read os.getenv

from data_cleaning_eda_crew.crew import DataCleaningEdaCrew  # noqa: E402  (must follow load_dotenv)


def run() -> None:
    print("Starting Automated Data Cleaning & EDA crew...\n")
    # "data_file" is exposed as a crew input (see tasks.yaml) so a deployed run
    # can point at an uploaded dataset's URL; locally it falls back to .env.
    inputs = {"data_file": os.getenv("DATA_FILE", "")}
    result = DataCleaningEdaCrew().crew().kickoff(inputs=inputs)

    print("\n" + "=" * 70)
    print("CREW FINISHED")
    print("=" * 70)
    out = Path("output")
    report = out / "reports" / "executive_summary.md"
    print(f"Executive summary : {report if report.exists() else '(not written)'}")
    print(f"Figures           : {out / 'figures'}")
    print(f"Cleaned CSVs       : {out / 'cleaned'}")
    print("\n--- Executive summary preview ---\n")
    print(str(result)[:2000])


if __name__ == "__main__":
    run()
