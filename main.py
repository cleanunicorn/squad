
import os
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

os.environ["OTEL_SDK_DISABLED"] = "true"

llm = Ollama(
    model = "lexi-llama3:8b",
    base_url = "http://localhost:11434")


# Define your agents with roles and goals
manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
    allow_delegation=True,
    llm=llm,
    verbose=True,
)

generator = Agent(
    role='Genius Solver',
    goal='Solve problems',
    backstory="""
    You are an expert at solving mathematical problems.
    Explain how the problems are solved and make things very clear with examples.
    Make sure to generate a correct solution.
    Output markdown.
    """,
    verbose=True,
    allow_delegation=False,
    llm=llm
)

checker = Agent(
    role='Expert Verifier',
    goal='Thoroughly evaluate and verify the accuracy of the problems and the solutions',
    backstory="""
    As the 'Expert Verifier', your primary responsibility is to ensure the correctness of problems and solutions. You meticulously check calculations, validate consistency of problems, thought process and solutions, and ensure the use of accurate and up-to-date real-world data. Your work ensures that each solution not only adheres to mathematical integrity but also aligns with the laws of nature and practical realities. Your expertise lies in spotting errors, suggesting improvements, and maintaining the highest standards of verification.
    """,
    llm=llm,
    verbose=True,
    allow_delegation=False
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
task1 = Task(
    description="Solve the following logic puzzle: Anne baked 60 cookies. Bob eats 10 cookies. Anne places the rest of the cookies in 50 jars. Bob gives Anne 5 cookies. How many cookies are in each jar?",
    expected_output="""
    A solution to the puzzle.
    """,
    agent=generator,
)

task2 = Task(
    description="Verify the correctness and plausibility of the solution provided for the puzzle.",
    expected_output="""
    A detailed verification of the solution, ensuring it adheres to logical reasoning and the constraints provided. The verification must confirm that the solution is consistent.
    """,
    agent=checker,
)

task3 = Task(
    description="Provide a concise summary of the solution's verification results.",
    expected_output="""
    A description of the solution and the final answer.
    A brief, clear summary highlighting the key points of the solution and verification.""",
    agent=summarizer,
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[generator, checker, summarizer],
  tasks=[task1, task2, task3],
  verbose=2, # You can set it to 1 or 2 to different logging levels
  process = Process.hierarchical,
  manager_agent=manager,
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
