import click
from dotenv import load_dotenv

# from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import CodeInterpreterTool
import os

import agents
import agents.manager
from llm import ollama, openai

# Load environment variables
load_dotenv()

# Disable telemetry in langchain
os.environ["OTEL_SDK_DISABLED"] = "true"


# CLI Commands
@click.command()
@click.argument(
    "tasks",
    type=int,
    default=1,
)
@click.option("--llm", default="ollama", show_default=True, type=str)
def cli(tasks, llm):
    """
    Command line interface to set up and solve tasks
    """

    # Select LLM to use
    if str.lower(llm) == "ollama":
        llm = ollama.llm()
    elif str.lower(llm) == "openai":
        llm = openai.llm()

    # Agents
    agent_solver = Agent(
        role="Solver",
        goal="Solve tasks",
        backstory="""
        As the 'Solver', your primary responsibility is to solve the task.
        You always spend a few sentences explaining background context, assumptions, and step-by-step thinking before you answer a question.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        # tools=[search, CodeInterpreterTool()],
    )

    # Tasks
    task_list = []

    # Read tasks
    for i in range(tasks):
        click.echo(f"Set up task #{i + 1}")
        task = Task(
            description=click.prompt("Description", default="The color of the sky"),
            expected_output=click.prompt(
                "Expected output", default="A word representing the color"
            ),
            agent=agent_solver,
        )
        task_list.append(task)

    # Select agents
    for i in range(tasks):
        pass

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[agent_solver],
        tasks=task_list,
        verbose=2,  # You can set it to 1 or 2 to different logging levels
        process=Process.hierarchical,
        manager_agent=agents.manager.manager(llm=llm),
    )

    # Get your crew to work!
    result = crew.kickoff()

    print("######################")
    print(result)


if __name__ == "__main__":
    cli()
