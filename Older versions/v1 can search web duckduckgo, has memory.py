import os
import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver  # <--- NEW: Import Memory
from langchain_core.messages import HumanMessage, AIMessage

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import API_KEY
os.environ["GOOGLE_API_KEY"] = API_KEY

# 1. THE BRAIN
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# 2. THE BODY
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# 3. THE MEMORY (New!)
# This saves the conversation state in RAM
memory = MemorySaver()

# 4. THE AGENT
# We add the 'checkpointer' here so it knows how to save history
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

def print_agent_output(content):
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and 'text' in block:
                print(f"💡 Agent: {block['text']}")
    else:
        print(f"💡 Agent: {content}")

def run_chat():
    print("🤖 Agent is ready! (Type 'quit' to exit)")
    print("-" * 30)
    
    # We need a thread_id so the agent knows WHO it is talking to
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        state = {"messages": [HumanMessage(content=user_input)]}
        
        # We pass 'config' here so it reloads the correct memory
        for event in agent_executor.stream(state, config=config, stream_mode="values"):
            message = event["messages"][-1]
            
            if isinstance(message, AIMessage):
                if message.tool_calls:
                    print(f"   🔎 Searching: {message.tool_calls[0]['args'].get('query')}")
                elif message.content:
                    print_agent_output(message.content)
        
        print("-" * 30)

if __name__ == "__main__":
    run_chat()
