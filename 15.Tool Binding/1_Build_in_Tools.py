"""

# General Tool Wrapper
from langchain.agents import Tool

# LLMs and Agents
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

# Search & Knowledge Tools
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun           # 1. Web search
from langchain.tools.wikipedia.tool import WikipediaQueryRun             # 2. Wikipedia
from langchain.tools.arxiv.tool import ArxivQueryRun                     # 3. Research papers
from langchain.tools.wolfram_alpha.tool import WolframAlphaQueryRun     # 4. Math & Science (needs API key)

# Python Tools
from langchain.tools.python.tool import PythonREPLTool                   # 5. Python REPL
from langchain.tools.shell.tool import ShellTool                         # 6. Shell command execution
from langchain.tools.jira.tool import JiraAction                         # 7. Jira integration

# File & Code Tools
from langchain.tools.file_management.read import ReadFileTool            # 8. Read files
from langchain.tools.file_management.write import WriteFileTool          # 9. Write files
from langchain.tools.file_management.list_dir import ListDirectoryTool   # 10. List files in directory
from langchain.tools.file_management.delete import DeleteFileTool        # 11. Delete files

# Web Requests
from langchain.tools.requests.tool import (
    RequestsGetTool,                                                     # 12. GET request
    RequestsPostTool,                                                    # 13. POST request
) 

# Developer Tools
from langchain.tools.openapi.tool import OpenAPISpecTool                 # 14. OpenAPI tool
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool       # 15. SQL query
from langchain.tools.vectorstore.tool import VectorStoreQATool           # 16. Vector search

# Calendar & Email Tools
from langchain.tools.gmail_search.tool import GmailSearchTool            # 17. Gmail search (Google auth needed)
from langchain.tools.google_calendar.tool import GoogleCalendarCreateTool # 18. Google Calendar create

# Utility Tools
from langchain.tools.human.tool import HumanInputRun                     # 19. Ask human
from langchain.tools.sleep.tool import SleepTool                         # 20. Wait/pause


"""




















from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke("what is ipl?" )
print(result)
