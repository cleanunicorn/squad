import os
from dotenv import load_dotenv

# from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import CodeInterpreterTool
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# Disable telemetry in langchain
os.environ["OTEL_SDK_DISABLED"] = "true"

# Search tool
search = DuckDuckGoSearchRun()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# llm = ChatOllama(
#     model = "codestral",
#     temperature = 0.4,
#     timeout=None,
#     # num_ctx=32768,
#     num_predict=None, # -1 infinite, -2 fill context
#     base_url = "http://localhost:11434",
# )

# Define your agents with roles and goals
manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory="""
    You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success.
    Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.
    """,
    allow_delegation=True,
    # max_iter=3,
    llm=llm,
    verbose=True,
)

generator = Agent(
    role="Solver",
    goal="Solve tasks",
    backstory="""
    As the 'Solver', your primary responsibility is to solve the task.
    You always spend a few sentences explaining background context, assumptions, and step-by-step thinking before you answer a question. 
    """,
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[search, CodeInterpreterTool()],
)

summarizer = Agent(
    role="Expert Summarizer",
    goal="To extract the core, most important ideas from text and present them in a concise and clear manner.",
    backstory="An AI designed to distill complex information into its essential elements, making it accessible and easy to understand.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# Create tasks for your agents

## Puzzle
# Anne baked 60 cookies. Bob eats 10 cookies. Anne places the rest of the cookies in 50 jars. Bob gives Anne 5 cookies. How many cookies are in each jar?

## Puzzle
# John is in the attic.
# He picks up a glass and walks to the basement.
# He puts a key in the glass and then walks to the living room.
# He turns the glass upside down, then walks to the kitchen.
# He moves the glass to the pantry and walks to the garden.
# Question: Where is the key?

task1 = Task(
    description="""
    Based on the local `main.py` Python script, create a CLI tool that accepts each task and outputs the result.
    """,
    expected_output="""
    A correct solution.
    """,
    agent=generator,
)

task2 = Task(
    description="Provide a concise summary of the solution's verification results.",
    expected_output="""
    A description of the solution and the final answer.
    A brief, clear summary highlighting the key points of the solution and verification.""",
    agent=summarizer,
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[generator, summarizer],
    tasks=[task1],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
    process=Process.hierarchical,
    manager_agent=manager,
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
