import os
import asyncio
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

# --- CORRECT IMPORTS ---
# 1. We import the raw Playwright library to launch it manually
from playwright.async_api import async_playwright
# 2. We still use the Toolkit to give the tools to the agent
from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit

from config import API_KEY
os.environ["GOOGLE_API_KEY"] = API_KEY

# 1. THE BRAIN
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

async def main():
    print("🔌 Launching Browser (Async Mode)...")
    
    # --- MANUAL BROWSER LAUNCH (Fixes the Loop Error) ---
    async with async_playwright() as p:
        # Launch the browser manually (Chromium is the engine for Chrome)
        browser = await p.chromium.launch(headless=False)
        
        # Initialize the toolkit with our manually created browser
        browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        tools = browser_toolkit.get_tools()

        # 3. THE MEMORY
        memory = MemorySaver()

        # 4. THE AGENT
        agent_executor = create_react_agent(llm, tools, checkpointer=memory)

        print("🤖 Browser Agent is ready! (Type 'quit' to exit)")
        print("-" * 30)
        
        config = {"configurable": {"thread_id": "1"}}

        while True:
            try:
                # Non-blocking input
                user_input = await asyncio.to_thread(input, "You: ")
            except KeyboardInterrupt:
                break

            if user_input.lower() in ["quit", "exit"]:
                break

            state = {"messages": [HumanMessage(content=user_input)]}
            
            try:
                # Run the agent loop
                async for event in agent_executor.astream(state, config=config, stream_mode="values"):
                    message = event["messages"][-1]
                    
                    if isinstance(message, AIMessage):
                        if message.tool_calls:
                            tool_name = message.tool_calls[0]['name']
                            args = message.tool_calls[0]['args']
                            print(f"   🔎 Agent is using tool '{tool_name}' with args: {args}")
                        elif message.content:
                            # Print final answer cleanly
                            if isinstance(message.content, list):
                                for block in message.content:
                                    if isinstance(block, dict) and 'text' in block:
                                        print(f"💡 Agent: {block['text']}")
                            else:
                                print(f"💡 Agent: {message.content}")
                                
            except Exception as e:
                print(f"❌ Error during execution: {e}")
            
            print("-" * 30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
