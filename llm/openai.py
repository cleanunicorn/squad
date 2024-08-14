from langchain_openai import ChatOpenAI


def llm(model="gpt-4o-mini", temperature=0.5):
    """
    Creates a chat interface using OpenAI's large language models.

    Args:
        model (str): The name of the LLM to use (default: "gpt-4o-mini").
        temperature (float): The temperature parameter for generating text (default: 0.5).

    Returns:
        ChatOpenAI: A chat interface instance.
    """

    llm_tool = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    return llm_tool
