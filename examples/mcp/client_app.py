# client_app.py
import asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


async def run_agent():
    # Define server parameters for the stdio connection
    server_params = StdioServerParameters(
        command="python",
        # Update to the full absolute path to your math_server.py file
        args=["math_server.py"],
    )

    # Initialize the model (e.g., GPT-4o)
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    # Establish the client connection and session
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection to discover tools
            await session.initialize()

            # Get tools exposed by the MCP server
            tools = await load_mcp_tools(session)

            # Create the LangGraph agent with the loaded tools
            agent = create_react_agent(model, tools)

            # Invoke the agent
            response = await agent.ainvoke({"messages": [("user", "What's (3 + 5) x 12?")]})
            print(response['messages'][-1].content)


# Run the asynchronous function
if __name__ == "__main__":
    asyncio.run(run_agent())
