from . import base


def manager(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory="""
        You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success.
        Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.
        """,
    llm=None,
):
    """
    This function creates a Manager agent.
    """

    return base.base_agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
    )
