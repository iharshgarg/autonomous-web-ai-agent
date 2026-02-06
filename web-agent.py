import os
import asyncio
import json
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

# --- IMPORTS ---
from langchain_core.tools import tool
from playwright.async_api import async_playwright

from config import API_KEY
os.environ["GOOGLE_API_KEY"] = API_KEY

# 1. THE BRAIN
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

async def main():
    print("🔌 Launching Crash-Safe Browser Agent...")
    
    async with async_playwright() as p:
        # Launch Browser
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # --- CUSTOM ROBUST TOOLS ---
        # These tools wrap everything in try/except blocks so the agent NEVER crashes.

        @tool
        async def safe_navigate(url: str):
            """Navigates to a URL safely."""
            try:
                # Ensure URL has protocol
                if not url.startswith("http"):
                    url = "https://" + url
                await page.goto(url, timeout=15000)
                return f"✅ Navigated to {url}"
            except Exception as e:
                return f"❌ Error navigating: {e}"

        @tool
        async def safe_fill_text(selector: str, text: str):
            """Fills a text field safely. selector is a CSS selector."""
            try:
                await page.wait_for_selector(selector, timeout=3000)
                await page.fill(selector, text)
                return f"✅ Typed '{text}' into '{selector}'"
            except Exception as e:
                return f"❌ Error typing: {e}"

        @tool
        async def safe_click(selector: str):
            """
            Tries to click an element using multiple methods (Standard -> Force -> JS).
            Use this for buttons, links, etc.
            """
            try:
                # 1. Wait for it
                try:
                    await page.wait_for_selector(selector, state="attached", timeout=3000)
                except:
                    pass # Keep going even if wait fails, maybe JS can find it

                # 2. Try Standard Click
                try:
                    await page.click(selector, timeout=1000)
                    return f"✅ Clicked '{selector}' (Standard)"
                except:
                    pass

                # 3. Try Force Click
                try:
                    await page.click(selector, force=True, timeout=1000)
                    return f"✅ Clicked '{selector}' (Force)"
                except:
                    pass

                # 4. Try JavaScript Click (The nuclear option)
                await page.evaluate(f"document.querySelector('{selector}').click()")
                return f"✅ Clicked '{selector}' (JS)"

            except Exception as e:
                return f"❌ Failed to click '{selector}': {e}"

        @tool
        async def safe_get_elements(selector: str):
            """
            Safely finds elements and returns their key attributes (id, name, text).
            Good for exploring the page to find button IDs.
            """
            try:
                # Validate selector to prevent crash
                if ":contains" in selector:
                    return "❌ Error: ':contains' is not valid CSS. Use standard CSS or ask for all buttons."
                
                elements = await page.query_selector_all(selector)
                results = []
                for el in elements[:10]: # Limit to 10 to save tokens
                    # Extract useful info
                    attrs = await page.evaluate("""(el) => {
                        return {
                            tag: el.tagName,
                            id: el.id,
                            text: el.innerText.slice(0, 50),
                            name: el.getAttribute('name'),
                            href: el.getAttribute('href')
                        }
                    }""", el)
                    results.append(attrs)
                return json.dumps(results)
            except Exception as e:
                return f"❌ Error reading elements: {e}"

        @tool
        async def safe_read_page():
            """Reads the visible text on the page."""
            try:
                content = await page.inner_text("body")
                return content[:2000] # Limit to avoid token overflow
            except Exception as e:
                return f"❌ Error reading page: {e}"

        @tool
        async def safe_find_text(text: str):
            """
            Finds elements containing specific text (e.g., 'Logout', 'Submit').
            Returns the selector you can use to click it.
            """
            try:
                # We use Playwright's text selector engine
                # This finds ANY tag containing the text
                selector = f"text={text}" 
                elements = await page.query_selector_all(selector)
                
                if not elements:
                    return f"❌ No elements found containing text '{text}'"
                
                results = []
                for i, el in enumerate(elements):
                    # We try to get a stable ID or Class, otherwise we suggest the text selector
                    attrs = await page.evaluate("""(el) => {
                        return {
                            tag: el.tagName,
                            id: el.id,
                            class: el.className,
                            visible: (el.offsetWidth > 0 && el.offsetHeight > 0)
                        }
                    }""", el)
                    
                    # We construct a usable selector for the agent
                    if attrs['id']:
                        usable_selector = f"#{attrs['id']}"
                    else:
                        usable_selector = f"text={text} >> nth={i}"
                        
                    results.append({
                        "found_text": text,
                        "tag": attrs['tag'],
                        "suggested_selector": usable_selector,
                        "is_visible": attrs['visible']
                    })
                
                return json.dumps(results)
            except Exception as e:
                return f"❌ Error finding text: {e}"

        # Combine our SAFE tools
        tools = [safe_navigate, safe_fill_text, safe_click, safe_get_elements, safe_read_page, safe_find_text]

        # 3. THE MEMORY
        memory = MemorySaver()

        # 4. THE AGENT
        agent_executor = create_react_agent(llm, tools, checkpointer=memory)

        print("🤖 Ready! (Type 'quit' to exit)")
        print("-" * 30)
        
        config = {"configurable": {"thread_id": "1"}}

        while True:
            try:
                user_input = await asyncio.to_thread(input, "You: ")
            except KeyboardInterrupt:
                break

            if user_input.lower() in ["quit", "exit"]:
                break

            state = {"messages": [HumanMessage(content=user_input)]}
            
            try:
                async for event in agent_executor.astream(state, config=config, stream_mode="values"):
                    message = event["messages"][-1]
                    
                    if isinstance(message, AIMessage):
                        if message.tool_calls:
                            tool_name = message.tool_calls[0]['name']
                            args = message.tool_calls[0]['args']
                            print(f"   🔎 Agent is calling '{tool_name}' with {args}")
                        elif message.content:
                            # Clean output
                            if isinstance(message.content, list):
                                for block in message.content:
                                    if isinstance(block, dict) and 'text' in block:
                                        print(f"💡 Agent: {block['text']}")
                            else:
                                print(f"💡 Agent: {message.content}")
                                
            except Exception as e:
                print(f"⚠️ Agent Logic Error: {e}")
                print("   (Don't worry, the browser is still alive!)")
            
            print("-" * 30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
