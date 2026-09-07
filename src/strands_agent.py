"""
Strands Agent Module
Local AI agent using Strands SDK with Amazon Bedrock for inference
and AgentCore Memory for long-term personalized memory.

Provides memory recall, storage, web search, and tool execution
capabilities for the InMoov robot voice assistant.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StrandsAgent:
    """Local Strands Agent for memory recall, reasoning, and tool execution.

    Uses Amazon Bedrock for LLM inference and AgentCore Memory for
    per-person long-term memory storage and retrieval.
    """

    # Supported LTM strategies
    LTM_STRATEGIES = ["session_summaries", "user_preferences", "semantic_facts"]

    # Default tool execution timeout (seconds)
    DEFAULT_TOOL_TIMEOUT = 5.0

    def __init__(self, memory_resource_id: str, region: str = "us-east-1"):
        """Initialize the Strands Agent.

        Args:
            memory_resource_id: The AgentCore Memory resource ID for storing/retrieving memories.
            region: AWS region for Bedrock and AgentCore services.
        """
        self.memory_resource_id = memory_resource_id
        self.region = region
        self._agent = None
        self._memory_client = None
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="StrandsAgent")

    def initialize(self) -> bool:
        """Initialize Strands agent with tools and Bedrock model.

        Sets up the Strands agent with Amazon Bedrock as the LLM provider
        and configures tools for memory recall and web search.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            from strands import Agent
            from strands.models.bedrock import BedrockModel
            from strands_tools import http_request

            # Set up Bedrock model
            model = BedrockModel(
                model_id="anthropic.claude-sonnet-4-20250514",
                region_name=self.region
            )

            # Set up AgentCore Memory client
            self._memory_client = self._create_memory_client()

            # Create the agent with tools
            self._agent = Agent(
                model=model,
                tools=[http_request],
                system_prompt=(
                    "You are a helpful assistant for the InMoov robot. "
                    "You help recall memories about people, search the web for information, "
                    "and provide context-aware responses. Be concise and friendly."
                )
            )

            self._initialized = True
            logger.info(f"Strands Agent initialized successfully (region={self.region})")
            return True

        except ImportError as e:
            logger.error(f"Strands Agent initialization failed - missing dependency: {e}")
            self._initialized = False
            return False
        except Exception as e:
            logger.error(f"Strands Agent initialization failed: {e}")
            self._initialized = False
            return False

    def _create_memory_client(self):
        """Create the AgentCore Memory data plane client.

        Returns:
            Memory client instance, or None if creation fails.
        """
        if not self.memory_resource_id:
            logger.warning("No memory_resource_id configured — memory operations will be unavailable")
            return None

        try:
            import boto3

            # AgentCore Memory uses 'bedrock-agentcore' for data plane operations
            # (create_event, retrieve memories, etc.)
            client = boto3.client(
                "bedrock-agentcore",
                region_name=self.region
            )
            return client
        except Exception as e:
            logger.warning(f"Failed to create AgentCore Memory client: {e}")
            return None

    @property
    def is_initialized(self) -> bool:
        """Check if the agent has been successfully initialized."""
        return self._initialized

    # --- Memory Operations ---

    def recall_memory(self, person_name: str, query: Optional[str] = None) -> str:
        """Retrieve memories for a person from AgentCore Memory.

        Scoped by actorId = person_name using namespace paths. Returns formatted
        memory context or empty string if memory is unavailable.

        Args:
            person_name: The person's name (used as actorId for scoping).
            query: Optional specific memory query to filter results.

        Returns:
            Formatted string of relevant memories, or empty string on failure.
        """
        if not self._memory_client:
            logger.warning("Memory client not available — skipping recall")
            return ""

        if not person_name:
            return ""

        try:
            # Use retrieve_memory_records to search across all strategies for this person
            # The namespace path scopes results to this person's memories
            search_query = query if query else f"What do I know about {person_name}?"

            response = self._memory_client.retrieve_memory_records(
                memoryId=self.memory_resource_id,
                namespacePath=f"/users/{person_name}/",
                searchCriteria={
                    "searchQuery": search_query,
                    "topK": 10
                }
            )

            # Also search summaries namespace
            try:
                summary_response = self._memory_client.retrieve_memory_records(
                    memoryId=self.memory_resource_id,
                    namespacePath=f"/summaries/{person_name}/",
                    searchCriteria={
                        "searchQuery": search_query,
                        "topK": 5
                    }
                )
                summaries = summary_response.get("memoryRecordSummaries", [])
            except Exception:
                summaries = []

            # Parse and format the memory records
            records = response.get("memoryRecordSummaries", [])
            all_records = records + summaries

            if not all_records:
                return ""

            formatted_parts = []
            for record in all_records:
                content = record.get("content", {})
                # Content can be a dict with 'text' key or similar structure
                if isinstance(content, dict):
                    text = content.get("text", str(content))
                else:
                    text = str(content)
                if text:
                    formatted_parts.append(text)

            return "\n".join(formatted_parts)

        except Exception as e:
            logger.warning(f"Memory recall failed for '{person_name}': {e}")
            return ""

    def store_memory(self, person_name: str, content: str,
                     strategy: str = "session_summaries") -> bool:
        """Store a memory entry for a person in AgentCore Memory.

        Uses create_event to write conversational data that the LTM strategies
        will process into long-term memory records.

        Args:
            person_name: The person's name (used as actorId for scoping).
            content: The memory content to store.
            strategy: LTM strategy type (session_summaries, user_preferences, semantic_facts).

        Returns:
            True if storage succeeded, False otherwise.
        """
        if not self._memory_client:
            logger.warning("Memory client not available — skipping store")
            return False

        if not person_name or not content:
            return False

        if strategy not in self.LTM_STRATEGIES:
            logger.warning(f"Unknown LTM strategy '{strategy}', using 'session_summaries'")
            strategy = "session_summaries"

        try:
            from datetime import datetime
            import uuid

            # Store as a conversational event — the LTM strategies will
            # automatically extract relevant information into memory records
            session_id = f"session_{person_name}_{int(time.time())}"

            self._memory_client.create_event(
                memoryId=self.memory_resource_id,
                actorId=person_name,
                sessionId=session_id,
                eventTimestamp=datetime.now(),
                payload=[
                    {
                        "conversational": {
                            "content": {"text": content},
                            "role": "ASSISTANT"
                        }
                    }
                ]
            )
            logger.debug(f"Stored memory event for '{person_name}' (strategy={strategy})")
            return True

        except Exception as e:
            logger.warning(f"Memory store failed for '{person_name}': {e}")
            return False

    def get_greeting_context(self, person_name: str, emotion: str) -> str:
        """Retrieve memory context formatted for greeting injection.

        Combines recalled memories with the current emotional state to
        produce a context string suitable for injecting into a greeting prompt.

        Args:
            person_name: The recognized person's name.
            emotion: The current detected emotional state.

        Returns:
            Formatted greeting context string, or minimal context on failure.
        """
        memories = self.recall_memory(person_name)

        context_parts = []

        if memories:
            context_parts.append(f"What you remember about {person_name}:")
            context_parts.append(memories)

        if emotion and emotion != "neutral":
            context_parts.append(f"Current mood: {person_name} appears to be {emotion}.")
        elif emotion == "neutral":
            context_parts.append(f"Current mood: {person_name} appears calm/neutral.")

        if not context_parts:
            return f"You are greeting {person_name}."

        return "\n".join(context_parts)

    # --- Tool Execution ---

    def search_web(self, query: str) -> str:
        """Perform web search and return summarized results.

        Uses the Strands web_search tool to find information.

        Args:
            query: The search query string.

        Returns:
            Search results as a formatted string, or error message on failure.
        """
        if not self._initialized or not self._agent:
            return "Web search is not available — agent not initialized."

        if not query or not query.strip():
            return "No search query provided."

        try:
            result = self.execute_tool("web_search", {"query": query})
            if "error" in result:
                return f"Search failed: {result.get('message', 'Unknown error')}"
            return result.get("result", "No results found.")
        except TimeoutError:
            return "Web search timed out. Please try again."
        except Exception as e:
            logger.warning(f"Web search failed for query '{query}': {e}")
            return "I couldn't search the web right now."

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a named tool with parameters.

        Runs the tool with a timeout to prevent blocking.

        Args:
            tool_name: Name of the tool to execute.
            parameters: Tool parameters as a dictionary.

        Returns:
            Result dictionary with tool output.

        Raises:
            TimeoutError: If execution exceeds the timeout (5 seconds).
        """
        if not self._initialized or not self._agent:
            return {"error": "not_initialized", "message": "Agent not initialized"}

        def _run_tool():
            # Use the Strands agent to execute the tool
            response = self._agent(
                f"Execute the {tool_name} tool with these parameters: {parameters}. "
                f"Return only the tool result, nothing else."
            )
            return {"result": str(response)}

        try:
            future = self._executor.submit(_run_tool)
            result = future.result(timeout=self.DEFAULT_TOOL_TIMEOUT)
            return result
        except FuturesTimeoutError:
            raise TimeoutError(
                f"Tool '{tool_name}' execution exceeded {self.DEFAULT_TOOL_TIMEOUT}s timeout"
            )
        except Exception as e:
            logger.error(f"Tool execution failed ({tool_name}): {e}")
            return {"error": "execution_failed", "message": str(e)}

    def shutdown(self) -> None:
        """Clean up resources used by the agent."""
        self._executor.shutdown(wait=False)
        self._initialized = False
        logger.info("Strands Agent shut down")
