# # import logging
# # import os
# # from dotenv import load_dotenv
# # from fastmcp import FastMCP
# # from starlette.requests import Request
# # from starlette.responses import JSONResponse
# # from modules.logger import get_logger
# # from qdrant_client import QdrantClient
# # from langchain_ollama import OllamaEmbeddings
# #
# #
# # QDRANT_URL = os.getenv("QDRANT_URL")
# # QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# #
# #
# # client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
# # embedder = OllamaEmbeddings(model="mxbai-embed-large")
# #
# # query = "What is gradient descent?"
# # query_vector = embedder.embed_query(query)
# # logger = get_logger("MCP_SERVER")
# #
# # os.system("clear")
# # load_dotenv()
# #
# # mcp = FastMCP("MyLac")
# #
# # @mcp.tool(
# #     name="lookup_textbooks",
# #     description="function to search from the knowledge base.",
# #     enabled=True,
# # )
# # def lookup_textbooks(query: str, top_k: int = 3) -> list:
# #     """
# #     Function to search from the ML knowledge base using Qdrant vector similarity.
# #     Args:
# #         query (str): The search query related to machine learning.
# #         top_k (int): The number of top results to return.
# #     Returns:
# #         list: Top matching chunks and their metadata.
# #     """
# #     logger.info(f"🔍 Searching Qdrant for query: {query}")
# #
# #     try:
# #         # Embed query
# #         query_emb = embedder.embed_query(query)
# #
# #         # Perform vector search
# #         search_results = client.query_points(
# #             collection_name=os.getenv("COLLECTION_NAME"),
# #             query=query_emb,
# #             limit=top_k,
# #         )
# #
# #         # Format the results for chatbot readability
# #         formatted_results = [
# #             {
# #                 "textbook": res.payload.get("textbook", "Unknown Textbook"),
# #                 "source_file": res.payload.get("source_file", "Unknown Source"),
# #                 "page": res.payload.get("page", "N/A"),
# #                 "score": res.score,
# #             }
# #             for res in search_results
# #         ]
# #
# #         logger.info(f"✅ Retrieved {len(formatted_results)} chunks from Qdrant")
# #         return formatted_results
# #
# #     except Exception as e:
# #         logger.error(f"❌ Error during Qdrant search: {e}")
# #         return [{"error": str(e)}]
# #
# #
# # if __name__ == "__main__":
# #     # try:
# #     #     logger.debug("MCP SERVER STARTED..")
# #     #     mcp.run(
# #     #         transport="streamable-http", log_level="WARNING", port=8000, host="0.0.0.0"
# #     #     )
# #     # except Exception as e:
# #     #     logger.error(f"❌ MCP server failed to start: {e}")
# #     logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
# #
# #     logger.debug("MCP SERVER STARTED..")
# #     mcp.run(transport="streamable-http",
# #             log_level="INFO",  # FastMCP own flag, optional
# #             port=8000, host="0.0.0.0")
#
#
#
# import os
# import logging
# from dotenv import load_dotenv
# from fastmcp import FastMCP
# from imdb.parser.sql.alchemyadapter import metadata
# from qdrant_client import QdrantClient
# from langchain_ollama import OllamaEmbeddings
# from modules.logger import get_logger
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
# import asyncio
# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# COLLECTION_NAME = os.getenv("COLLECTION_NAME")
#
# client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
# embedder = OllamaEmbeddings(model="mxbai-embed-large")
# logger = get_logger("MCP_SERVER")
# OUTPUT_PARSER = StrOutputParser()
# mcp = FastMCP("MyLac")
# #
# # @mcp.tool(
# #     name="lookup_textbooks",
# #     description="Search the ML knowledge base using vector similarity.",
# #     enabled=True,
# # )
# async def lookup_textbooks(query: str, top_k: int = 3) -> list:
#     '''
#     this is used whenever any Machine Learning related query is asked
#
#     '''
#     try:
#         query_emb = embedder.embed_query(query)
#         results = client.query_points(collection_name=COLLECTION_NAME, query=query_emb, limit=top_k)
#         points = results[0]
#
#         metadata=[
#             {
#                 "textbook": r.payload.get("textbook", "Unknown"),
#                 "source_file": r.payload.get("source_file", "Unknown"),
#                 "page": r.payload.get("page", "N/A"),
#                 "score": r.score,
#             }
#             for r in points ]
#
#     llm = await create_llm(
#         model_provider=os.getenv("MODEL_PROVIDER", "ollama"),
#         model_name=os.getenv("MODEL"),
#     )
#     msg = "Use this content to answer the query" + results  + "textbooks feild from "+ metadata + "as source "
#     system_msg = SystemMessage(content=prompts.SYSTEM_PROMPT + msg)
#
#     logger.info("Agent ready ✅")
#
#     while True:
#         query = input("🧑 You: ").strip()
#         # result = await client.call_tool("lookup_textbooks", {"query": query})
#
#         if query.lower() in ["exit", "quit", "q"]:
#             break
#         if not query:
#             continue
#
#         human_msg = HumanMessage(content=query)
#         chain = human_msg + system_msg | llm | OUTPUT_PARSER
#
#         try:
#             response = await chain.invoke({"messages": [system_msg, human_msg]})
#             parsed = OUTPUT_PARSER.invoke(response.get("messages")[-1])
#             print(f"🤖 {parsed}")
#         except Exception as e:
#             logger.error(f"BOT error: {e}")
#
#
#
# # if __name__ == "__main__":
# #     logging.basicConfig(level=logging.INFO)
# #     logger.info("🚀 Starting MCP server...")
# #     mcp.run(transport="streamable-http", log_level="INFO", port=8000, host="0.0.0.0")
# lookup_textbooks(query , top_k=3)
