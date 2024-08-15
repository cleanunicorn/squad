# Standard library imports
import os
from textwrap import dedent

# Third-party imports
import click
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import CodeInterpreterTool, WebsiteSearchTool, FileReadTool

# Local application imports
import agents
import agents.manager
from llm import ollama, openai


# Load environment variables
load_dotenv()

# Disable telemetry in langchain
os.environ["OTEL_SDK_DISABLED"] = "true"


def command_to_task(llm, command) -> str:
    """
    This is a function that generates a prompt to be used with a Large Language Model (LLM) based on a given command from a user.
    The function aims to create a clear and concise description of the context and purpose of the command, ensuring the LLM understands the desired outcome.
    It follows specific style guidelines such as being clear, concise, creative, and neutral. The generated prompt also focuses on the appropriate audience by using accessible language.

    Parameters:
    - llm (object): An instance of a Large Language Model that will be used to generate a response based on the provided prompt.
    - command (str): A string representing the command or request issued by the user for which the LLM needs to provide an appropriate response.

    Returns:
    - str: A string containing the generated prompt based on the input command and the LLM's ability to interpret it as per the style guidelines and audience consideration.
    """

    prompt = dedent(
        """
    ### CONTEXT ###
    You are an expert prompt engineer tasked with crafting prompts for large language models (LLMs). Your objective is to create a prompt that enables an LLM to effectively respond to a command issued by a human. The prompt you generate must prioritize clarity, conciseness, and creativity, while ensuring neutrality and avoiding bias or reliance on stereotypes.

    ### OBJECTIVE ###
    Your task is to develop this component:
    1. **DESCRIPTION:** A concise explanation of the context and purpose of the command, ensuring the LLM understands the desired outcome.

    The generated prompt should reflect a deep understanding of the command while maintaining clarity and precision in both structure and language.

    {command}

    ### FORMAT ###
    The generated prompt should follow the format demonstrated below. Consistency in tone, logic, and formatting is critical.

    ### EXAMPLES ###
    #### Example 1:
    **Input:** What is the color of the sky?

    **Output:**

    **### DESCRIPTION ###**  
    This prompt asks for a scientific explanation of the sky's color. The LLM should leverage its knowledge of atmospheric phenomena to provide a clear and factual response. The explanation should be concise and grounded in basic scientific principles related to the scattering of sunlight.

    ---

    #### Example 2:
    **Input:** Write a poem about autumn leaves.

    **Output:**

    **### DESCRIPTION ###**  
    This command requests a creative response in the form of a poem about autumn leaves. The LLM should focus on evoking the imagery and emotions associated with the changing colors, the passage of time, and the transition from summer to winter.

    ---

    ### STYLE GUIDELINES ###
    - **Clarity:** Ensure the prompts are clear and easy to understand.
    - **Conciseness:** Avoid unnecessary elaboration or repetition.
    - **Creativity:** Encourage the LLM to be imaginative, where appropriate.
    - **Neutrality:** Avoid introducing bias or relying on stereotypes.

    ### AUDIENCE ###
    The audience includes individuals seeking both factual and creative responses from the LLM. The language should be accessible, professional, and free from overly technical jargon.

    ### FINAL OUTPUT INSTRUCTIONS ###
    Once you have reviewed the command provided by the human, structure your response in accordance with the format outlined above, ensuring the component `DESCRIPTION` is well-formulated and tailored to the command.
    """
    )
    generated_prompt = llm.invoke(prompt.format(command=command)).content
    return generated_prompt


# CLI Commands
@click.command()
@click.argument(
    "tasks",
    type=int,
    default=1,
)
@click.option("--llm", default="ollama", show_default=True, type=str)
@click.option("--model", default="llama3.1", show_default=True, type=str)
def main(tasks, llm, model):
    """
    Command line interface to set up and solve tasks
    """

    # Select LLM to use
    if str.lower(llm) == "ollama":
        llm = ollama.llm(model=model)
    elif str.lower(llm) == "openai":
        if model == "llama3.1":
            model = "gpt-4o-mini"
        llm = openai.llm(model=model)

    # Setup tools
    search_tool = DuckDuckGoSearchRun()
    website_rag_tool = WebsiteSearchTool()
    file_read_tool = FileReadTool()

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
        # tools=[search_tool],
    )

    # Tasks
    task_list = []

    # Read tasks
    for i in range(tasks):
        click.echo(f"Set up task #{i + 1} of {tasks}")

        # Preprocess command
        command = click.prompt(
            "Prompt", default="What is the color of the sky? Let's think step by step"
        )
        generated_prompt = command_to_task(llm, command)

        task = Task(
            description=generated_prompt,
            expected_output="",
            agent=agent_solver,
        )
        task_list.append(task)

    # TODO: Select agents
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
    main()
