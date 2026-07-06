"""Crew definition: 3 agents (Profiler -> Code/Viz Developer -> Insights Writer)."""

from __future__ import annotations

import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, before_kickoff, crew, task

from data_cleaning_eda_crew.tools.eda_tools import (
    clean_dataset,
    profile_dataset,
    set_data_source,
    visualize_dataset,
)


def _build_llm() -> LLM:
    """Single shared LLM, configured from the .env file (Gemini via LiteLLM)."""
    return LLM(
        model=os.getenv("MODEL", "gemini/gemini-3.1-flash-lite"),
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2,
        # Route through LiteLLM (verified working with this key/model) instead of
        # crewai's native google-genai provider, which needs an extra dependency.
        is_litellm=True,
    )


@CrewBase
class DataCleaningEdaCrew:
    """Automated Data Cleaning & EDA crew."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ---- Hooks -------------------------------------------------------------- #
    @before_kickoff
    def load_data_source(self, inputs: dict) -> dict:
        """If the caller supplied a data_file (local path or http(s) URL),
        point the EDA tools at it instead of the DATA_FILE env default."""
        set_data_source(inputs.get("data_file", ""))
        return inputs

    # ---- Agents ----------------------------------------------------------- #
    @agent
    def data_profiler(self) -> Agent:
        return Agent(
            config=self.agents_config["data_profiler"],
            tools=[profile_dataset],
            llm=_build_llm(),
            verbose=True,
        )

    @agent
    def code_developer(self) -> Agent:
        return Agent(
            config=self.agents_config["code_developer"],
            tools=[clean_dataset, visualize_dataset],
            llm=_build_llm(),
            verbose=True,
        )

    @agent
    def insights_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["insights_summarizer"],
            llm=_build_llm(),
            verbose=True,
        )

    # ---- Tasks ------------------------------------------------------------ #
    @task
    def profiling_task(self) -> Task:
        return Task(config=self.tasks_config["profiling_task"])

    @task
    def engineering_task(self) -> Task:
        return Task(config=self.tasks_config["engineering_task"])

    @task
    def reporting_task(self) -> Task:
        return Task(config=self.tasks_config["reporting_task"])

    # ---- Crew ------------------------------------------------------------- #
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
