"""
A module for providing solver functionality.
"""

from . import base


def solver(
    role="Solver",
    goal="Solve tasks",
    backstory="""
    As the 'Solver', your primary responsibility is to solve the task.
    You always spend a few sentences explaining background context, assumptions, and step-by-step thinking before you answer a question.
    """,
    llm=None,
    allow_delegation=False,
    verbose=True,
):
    """
    This function generates a generic solver agent.
    """

    return base.base_agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        allow_delegation=allow_delegation,
        verbose=verbose,
    )
