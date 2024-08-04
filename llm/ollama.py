from langchain_ollama import ChatOllama


def llm(model="llama3.1", temperature=0.5):
    """
    Creates a chat interface using Ollama's large language models.

    Args:
        model (str): The name of the LLM to use (default: "llama3.1").
        temperature (float): The temperature parameter for generating text (default: 0.5).

    Returns:
        ChatOllama: A chat interface instance.
    """

    llm_tool = ChatOllama(
        model=model,
        temperature=temperature,
        timeout=None,
        # num_ctx=32768,
        num_predict=None,  # -1 infinite, -2 fill context
        base_url="http://localhost:11434",
    )

    return llm_tool
