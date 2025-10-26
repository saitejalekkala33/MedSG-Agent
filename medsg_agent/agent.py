"""
Agent wiring using LangChain's initialize_agent with OpenAI chat model.
"""
import os, json
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from medsg_agent.tools import get_tools  

def _load_env():
    candidates = [
        Path(__file__).with_name(".env"),                
        Path(__file__).resolve().parent.parent / ".env", 
        Path.cwd() / ".env",                             
    ]
    loaded_any = False
    for p in candidates:
        if p.exists():
            load_dotenv(p, override=False)
            loaded_any = True
    load_dotenv(override=False)
    return loaded_any

def build_agent(model_name: str = "gpt-4o-mini", temperature: float = 0.1):
    """
    Create a tool-using agent. Loads .env and passes api_key to ChatOpenAI.
    """
    _load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Put it in one of:\n"
            f" - {Path(__file__).with_name('.env')}\n"
            f" - {Path(__file__).resolve().parent.parent / '.env'}\n"
            f" - {Path.cwd() / '.env'}\n"
            "Or set it in the environment."
        )
    base_url = os.getenv("OPENAI_BASE_URL")  

    tools = get_tools()
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url if base_url else None,
    )
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,  
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent

if __name__ == "__main__":
    agent = build_agent()
    question = (
        "Given these two CT slices, find the difference and output bbox coordinates as JSON. "
        '{"image_a":"C:/cpp/Medical/human_samples/Tools/registered_Diff/CTPelvic1K_CT_LumbarSpine_npy_imgs_ori_casenum_0027_sliceid_219.png", '
        '"image_b":"C:/cpp/Medical/human_samples/Tools/registered_Diff/CTPelvic1K_CT_LumbarSpine_npy_imgs_casenum_0027_sliceid_219.png"}'
    )
    result = agent.run(question)
    print(result)
