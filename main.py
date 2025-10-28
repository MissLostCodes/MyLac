# # from langchain_core.prompts import ChatPromptTemplate
# #
# # prompt = ChatPromptTemplate.from_messages([
# #     ("system", "You are a helpful AI tutor. Use the retrieved context to answer."),
# #     ("human", "Question: {question}\n\nContext:\n{context}")
# # ])
# # from langchain_core.output_parsers import StrOutputParser
# #
# # output_parser = StrOutputParser()
# # from langchain_community.llms import Ollama
# # from langchain_community.vectorstores import Qdrant
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_core.prompts import ChatPromptTemplate
# #
# # # --- Setup ---
# # embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# # vectorstore = Qdrant(
# #     url="http://localhost:6333",
# #     collection_name="ml_textbooks",
# #     embeddings=embedding_model
# # )
# # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# # llm = Ollama(model="llama3")
# #
# # # --- Prompt ---
# # prompt = ChatPromptTemplate.from_messages([
# #     ("system", "You are an expert ML tutor. Answer using only the retrieved context."),
# #     ("human", "Question: {question}\n\nContext:\n{context}")
# # ])
# #
# # # --- Output Parser ---
# # output_parser = StrOutputParser()
# #
# # # --- Define the chain ---
# # rag_chain = (
# #     {
# #         "context": retriever,      # feeds retrieved text into the prompt
# #         "question": lambda x: x["question"]
# #     }
# #     | prompt
# #     | llm
# #     | output_parser
# # )
# # response = rag_chain.invoke({"question": "Explain overfitting in machine learning."})
# # print(response)
#
# # Query Agent without Login for 5 free queries
# import logging
# # import os
# # from typing import List, Dict, Any
# # import asyncio
# # from dotenv import load_dotenv
# # from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_mcp_adapters.client import MultiServerMCPClient
# # from langgraph.prebuilt import create_react_agent
# # from mcp.client.streamable_http import streamablehttp_client
# # from modules.logger import get_logger
# # from qdrant_client import QdrantClient
# # from modules.prompts import prompts
# # from modules.agent import create_llm
# # logger = get_logger("MAIN.PY")
# #
# # logger.info("Starting MCP CLIENT")
# # load_dotenv()
# #
# # OUTPUT_PARSER = StrOutputParser()
# # system_msg = SystemMessage(content=prompts.SYSTEM_PROMPT)
# # #user_msg = #============take input from user
# # MCP_TOOLS = {"ML_query_search": {
# #                         "url": os.getenv("SERVER_URL"),
# #                         "transport": "streamable_http",
# #                 }}
# # #
# # # stream_ctx = streamablehttp_client(os.getenv("SERVER_URL"))
# # # read, write, _ = stream_ctx.__aenter__()
# # client = MultiServerMCPClient(MCP_TOOLS)
# # tools =  client.get_tools()
# # llm = create_llm(
# #             model_provider=os.getenv("MODEL_PROVIDER"),
# #             model_name=os.getenv("MODEL"),
# #         )
# # agent = create_react_agent(model=llm, tools=tools)
# # logger.info("Bot working fine ! ")
# #
# #
# # async def invoke_agent( question: str):
# #     try:
# #         agent_response = await agent.ainvoke({"message": question })
# #         print(agent_response)
# #         parsed_response = OUTPUT_PARSER.invoke(agent_response.get("message")[-1])
# #         return parsed_response
# #     except Exception as e:
# #         return f"Agent error: {e}"
# #
# # async def main():
# #     try:
# #         async with streamablehttp_client(os.getenv("SERVER_URL")) as (read, write, _):
# #             client = MultiServerMCPClient(MCP_TOOLS)
# #             all_tools = await client.get_tools()
# #             logger.info(f" Loaded tool: {[t.name for t in all_tools]}")
# #
# #             llm = await create_llm(
# #                 model_provider=os.getenv("MODEL_PROVIDER"),
# #                 model_name=os.getenv("MODEL"),
# #             )
# #             agent = create_react_agent(model=llm, tools=all_tools)
# #
# #             while True:
# #                 logger.debug("\n" + "=" * 90 + "\n")
# #                 query = input("🧑 You: ").strip()
# #
# #                 if not query:
# #                     logger.info("Please enter a valid query")
# #                     continue
# #
# #                 if query.lower() in ["exit", "q", "quit"]:
# #                     logger.info("Exiting...")
# #                     break
# #
# #
# #                 system_msg = SystemMessage(content=prompts.SYSTEM_PROMPT)
# #                 human_msg =   HumanMessage(content=query)
# #                 final_messages = [system_msg] + human_msg
# #
# #                 try:
# #                     agent_response = await agent.ainvoke({"messages": final_messages})
# #                     parsed_response = OUTPUT_PARSER.invoke(
# #                         agent_response.get("messages")[-1]
# #                     )
# #                     logger.info(f"🤖: {parsed_response}")
# #
# #                 except Exception as e:
# #                     logger.error(f"❌ BOT error: {e}")
# #                     continue
# #     except KeyboardInterrupt:
# #         logger.error("Shutting down client...")
# #         logger.info("Exiting...")
# #     except Exception as e:
# #         import traceback
# #
# #         logger.error(f"❌ Fatal error: {e}")
# #         traceback.print_exc()
# #         logger.info("Exiting...")
# #
# #
# # if __name__ == "__main__":
# #     asyncio.run(main())
# # @app.get(
# #     "/chat/free",
# #     summary="Query agent without login",
# #     description="Query the agent module without login.Note: This endpoint is rate limited to 5 requests per 720 hours.",
# #     response_description="Agent response",
# #     response_class=JSONResponse,
# #     dependencies=[
# #         Depends(RateLimiter(times=5, hours=24))
# #     ],
# # )
# # async def free_query(query: str, request: Request):
# #     """
# #     Query the agent module without login.
# #     Note: This endpoint is rate limited to 5 requests per 720 hours.
# #
# #     Parameters:
# #         query (str): Query string.
# #         request : Injected by dependency.
# #
# #     Returns:
# #         dict: Agent response.
# #     """
# #     ip = request.client.host
# #     session_id = f"user_free_{ip}"
# #     response = await agent_mod.invoke_agent(session_id.strip(), query)
# #     return responsemain
#
# #
# # import os
# # import asyncio
# # from dotenv import load_dotenv
# # from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_mcp_adapters.client import MultiServerMCPClient
# # from langgraph.prebuilt import create_react_agent
# # from mcp.client.streamable_http import streamablehttp_client
# # from modules.logger import get_logger
# # from qdrant_client import QdrantClient
# # from modules.prompts import prompts
# # from modules.llm_init import create_llm
# # from fastmcp import Client, FastMCP
# # from langchain_mcp_adapters.tools import load_mcp_tools
# #
# # logger = get_logger("MAIN.PY")
# #
# # logger.info("Starting MCP CLIENT")
# # load_dotenv()
# # MODEL_PROVIDER="ollama"
# # MODEL="qwen3:4b"
# # OUTPUT_PARSER = StrOutputParser()
# # MCP_TOOLS = {
# #     "lookup_textbooks": {
# #         "url": os.getenv("SERVER_URL"),
# #         "transport": "streamable_http",
# #     }
# # }
# #
# #
# # # async def call_tool(name: str):
# # #     async with client:
# # #         result = await client.call_tool("greet", {"name": name})
# # #         print(result)
# # #
# # # asyncio.run(call_tool("Ford"))
# # async def main():
# #     # client = Client(os.getenv("SERVER_URL"))
# #     try:
# #         client = MultiServerMCPClient(MCP_TOOLS)
# #         async with client.session("lookup_textbooks") as session:
# #             tools = await load_mcp_tools(session)
# #         # async with client:
# #         #     # Initialize MCP client and tools
# #         #     # client = MultiServerMCPClient(MCP_TOOLS)
# #
# #             # all_tools = await client.load_mcp_tools()
# #             logger.info(f"Loaded tools: {[t.name for t in tools]}")
# #
# #             # Create LLM instance
# #             logger.info("-------------------LLM being created ----------------")
# #             llm = await create_llm(
# #                 model_provider=os.getenv("MODEL_PROVIDER"),
# #                 model_name=os.getenv("MODEL"),
# #             )
# #
# #             # Create LangChain agent
# #             agent = create_react_agent(model=llm, tools=tools)
# #             logger.info(f"Created react agent: {agent}")
# #             logger.info("Agent initialized successfully ✅")
# #
# #             # Chat loop
# #             while True:
# #                 logger.debug("\n" + "=" * 90 + "\n")
# #                 query = input("🧑 You: ").strip()
# #
# #                 if not query:
# #                     logger.info("Please enter a valid query")
# #                     continue
# #
# #                 if query.lower() in ["exit", "q", "quit"]:
# #                     logger.info("Exiting...")
# #                     break
# #
# #                 # Compose message stack
# #                 system_msg = SystemMessage(content=prompts.SYSTEM_PROMPT)
# #                 human_msg = HumanMessage(content=query)
# #                 final_messages = [system_msg, human_msg]
# #
# #                 try:
# #                     # Get response
# #                     agent_response = await agent.ainvoke({"messages": final_messages})
# #                     parsed_response = OUTPUT_PARSER.invoke(
# #                         agent_response.get("messages")[-1]
# #                     )
# #                     logger.info(f"🤖: {parsed_response}")
# #
# #                 except Exception as e:
# #                     logger.error(f"❌ BOT error: {e}")
# #                     continue
# #
# #     except KeyboardInterrupt:
# #         logger.error("Shutting down client...")
# #     except Exception as e:
# #         import traceback
# #
# #         logger.error(f"❌ Fatal error: {e}")
# #         traceback.print_exc()
# #     finally:
# #         logger.info("Exiting...")
# #
# #
# # if __name__ == "__main__":
# #     asyncio.run(main())
# import os
# import asyncio
# from dotenv import load_dotenv
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
# from modules.logger import get_logger
# from modules.prompts import prompts
# from modules.llm_init import create_llm
# from langchain_mcp_adapters.tools import load_mcp_tools
#
# load_dotenv()
# logger = get_logger("MAIN.PY")
#
# OUTPUT_PARSER = StrOutputParser()
# MCP_TOOLS = {
#     "lookup_textbooks": {
#         "url": os.getenv("SERVER_URL", "http://localhost:8000"),
#         "transport": "streamable_http",
#
#     }
# }
#
# async def main():
#     try:
#         client = MultiServerMCPClient(MCP_TOOLS)
#         async with client:
#             #tools = await load_mcp_tools(session)
#             #logger.info(f"Loaded tools: {[t.name for t in tools]}")
#
#
#             llm = await create_llm(
#                 model_provider=os.getenv("MODEL_PROVIDER", "ollama"),
#                 model_name=os.getenv("MODEL", "qwen3:4b"),
#             )
#             agent = create_react_agent(model=llm)
#             logger.info("Agent ready ✅")
#
#             while True:
#                 query = input("🧑 You: ").strip()
#                 result = await client.call_tool("lookup_textbooks", {"query": query})
#                 if query.lower() in ["exit", "quit", "q"]:
#                     break
#                 if not query:
#                     continue
#                 msg ="Use this content to answer the query" +  result
#                 system_msg = SystemMessage(content=prompts.SYSTEM_PROMPT+msg)
#                 human_msg = HumanMessage(content=query)
#
#                 try:
#                     response = await agent.ainvoke({"messages": [system_msg, human_msg]})
#                     parsed = OUTPUT_PARSER.invoke(response.get("messages")[-1])
#                     print(f"🤖 {parsed}")
#                 except Exception as e:
#                     logger.error(f"BOT error: {e}")
#
#     except Exception as e:
#         logger.error(f"Fatal error: {e}")
#
# if __name__ == "__main__":
#     asyncio.run(main())

#
import os
import asyncio
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from modules.logger import get_logger
from modules.prompts import prompts
from modules.llm_init import create_llm

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger("ML_AGENT")

# Environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Qdrant and embedding setup
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
embedder = OllamaEmbeddings(model="mxbai-embed-large")

# Output parser
OUTPUT_PARSER = StrOutputParser()

#
# async def lookup_textbooks(query: str, top_k: int = 3):
#     """
#     Searches the ML knowledge base using vector similarity and answers using an LLM.
#     """
#     try:
#         # Create embedding for query
#         query_emb = embedder.embed_query(query)
#
#         # Query Qdrant
#         results = client.query_points(
#             collection_name=COLLECTION_NAME,
#             query=query_emb,
#             limit=top_k
#         )
#
#
#         logger.info(f"Found  : {results}  for {query}")
#         points = results.points
#
#         metadata = [
#             {
#                 "textbook": r.payload.get("textbook", "Unknown"),
#                 "source_file": r.payload.get("source_file", "Unknown"),
#                 "page": r.payload.get("page", "N/A"),
#                 "score": r.score,
#             }
#             for r in points
#         ]
#
#         # Create LLM instance
#         llm = await create_llm(
#             model_provider=os.getenv("MODEL_PROVIDER", "ollama"),
#             model_name=os.getenv("MODEL"),
#         )
#
#         # Combine search results for the system message
#         msg = f"Use the following textbook content to answer the query: \n\n{results}\n\n"
#         system_msg = SystemMessage(content=prompts.SYSTEM_PROMPT + msg)
#
#         logger.info("Agent ready ✅")
#
#         while True:
#             query = input("🧑 You: ").strip()
#
#             if query.lower() in ["exit", "quit", "q"]:
#                 break
#             if not query:
#                 continue
#
#             human_msg = HumanMessage(content=query)
#
#             try:
#                 # Run LLM
#                 response = await llm.ainvoke([system_msg, human_msg])
#                 parsed = OUTPUT_PARSER.invoke(response)
#                 print(f"\n🤖 {parsed}+{metadata}\n")
#             except Exception as e:
#                 logger.error(f"BOT error: {e}")
#
#     except Exception as e:
#         logger.error(f"Error in lookup_textbooks: {e}")
#
#
# if __name__ == "__main__":
#     query = input("🔍 Enter your initial ML query: ").strip()
#     asyncio.run(lookup_textbooks(query, top_k=3))
async def start_conversation(top_k: int = 3):
    """
    Starts the conversation loop directly without asking for an initial query.
    """
    try:
        # Create LLM instance once
        llm = await create_llm(
            model_provider=os.getenv("MODEL_PROVIDER", "ollama"),
            model_name=os.getenv("MODEL"),
        )

        logger.info("Agent ready ✅")

        while True:
            query = input("🧑 You: ").strip()

            if query.lower() in ["exit", "quit", "q"]:
                print("👋 Conversation ended.")
                break
            if not query:
                continue

            # Create embedding for query
            query_emb = embedder.embed_query(query)

            # Query Qdrant
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_emb,
                limit=top_k
            )

            points = results.points
            metadata = [
                {
                    "textbook": r.payload.get("textbook", "Unknown"),
                    "source_file": r.payload.get("source_file", "Unknown"),
                    "page": r.payload.get("page", "N/A"),
                    "score": r.score,
                }
                for r in points
            ]
 #////////////////   thwse fucking points to actual eng
            # Combine search results for the system message
            msg = f"Use the following textbook content to answer the query: \n\n{results.retrieve}\n\n"
            system_msg = SystemMessage(content=prompts.SYSTEM_PROMPT + msg)
            human_msg = HumanMessage(content=query)

            try:
                response = await llm.ainvoke([system_msg, human_msg])
                parsed = OUTPUT_PARSER.invoke(response)
                print(f"\n🤖 {parsed}\n📘 Context: {metadata}\n")
            except Exception as e:
                logger.error(f"BOT error: {e}")

    except Exception as e:
        logger.error(f"Error in start_conversation: {e}")


if __name__ == "__main__":
    asyncio.run(start_conversation(top_k=3))
