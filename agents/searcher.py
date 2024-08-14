from . import base
from langchain_community.tools import DuckDuckGoSearchRun


def searcher(
    role="Searcher",
    goal="Search for information on the internet",
    backstory="""
    The Searcher is an AI-powered research assistant designed to scour the vast expanse of the internet for relevant information on a given topic. Its primary function is to retrieve and provide users with accurate, up-to-date knowledge on a wide range of subjects.
    """,
    llm=None,
    allow_delegation=False,
    verbose=True,
):
    """
    This function generates a searcher agent.
    """

    # Search tool
    search = DuckDuckGoSearchRun()

    return base.base_agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        allow_delegation=allow_delegation,
        verbose=verbose,
        tools=[search],
    )
