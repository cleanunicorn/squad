from crewai import Agent


def base_agent(
    role="Solver",
    goal="Solve tasks",
    backstory="""
    As the 'Solver', your primary responsibility is to solve the task.
    You always spend a few sentences explaining background context, assumptions, and step-by-step thinking before you answer a question.
    """,
    llm=None,
    allow_delegation=False,
    verbose=True,
    tools=None,
):
    """
    Base agent
    """

    if llm is None:
        print("The llm can't be None")
        return

    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        allow_delegation=allow_delegation,
        verbose=verbose,
        tools=tools,
    )

    return agent
