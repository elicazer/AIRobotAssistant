"""
Tool Handler Module
Handles Nova Sonic 2 tool use: registration, execution, and response routing.

Manages tool definitions for recall_memory, get_emotion, and search_web,
routes tool calls to the appropriate handler (Strands Agent or Emotion Detector),
and enforces execution timeouts.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, List, Optional

from emotion_detector import EmotionDetector
from strands_agent import StrandsAgent

logger = logging.getLogger(__name__)


class ToolHandler:
    """Handles Nova Sonic 2 tool use: registration, execution, and response.

    Routes tool calls to the appropriate backend:
    - recall_memory → Strands Agent (AgentCore Memory)
    - search_web → Strands Agent (web search)
    - get_emotion → Emotion Detector (direct read)
    """

    TOOL_DEFINITIONS: List[Dict[str, Any]] = [
        {
            "toolSpec": {
                "name": "recall_memory",
                "description": (
                    "Recall memories and past conversation context for the person "
                    "currently being spoken to"
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Optional specific memory query"
                            }
                        }
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "get_emotion",
                "description": (
                    "Get the current emotional state of the person being spoken to "
                    "based on facial expression analysis"
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "set_expression",
                "description": (
                    "Show an emotion on your own (the avatar's) face so the person "
                    "can see and read the feeling. Use this often and deliberately: "
                    "match your facial expression to what you're saying, and when "
                    "teaching or naming an emotion, set the matching expression so "
                    "the person can connect the word to the face. Call with "
                    "'neutral' to return to a calm resting face."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "emotion": {
                                "type": "string",
                                "description": (
                                    "One of: happy, sad, angry, relaxed, neutral "
                                    "(synonyms like calm, excited, upset are accepted)"
                                )
                            },
                            "intensity": {
                                "type": "number",
                                "description": "Optional expression strength 0.0-1.0 (default 1.0)"
                            }
                        },
                        "required": ["emotion"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "search_web",
                "description": "Search the web for current information to answer a question",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "enroll_face",
                "description": (
                    "Enroll the face of the person currently in front of the camera "
                    "so the robot can recognize them in the future. Call this when "
                    "someone tells you their name and you want to remember their face. "
                    "The camera will capture their face automatically."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The person's name to associate with their face"
                            }
                        },
                        "required": ["name"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "forget_person",
                "description": (
                    "Forget a person completely — removes their face enrollment and all "
                    "stored memories/conversation history. Use when someone asks you to "
                    "forget them or delete their data, or to correct a mistaken enrollment."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The person's name to forget"
                            }
                        },
                        "required": ["name"]
                    }
                }
            }
        }
    ]

    # Default timeout for tool execution (seconds)
    DEFAULT_TIMEOUT = 10.0

    def __init__(
        self,
        strands_agent: StrandsAgent,
        emotion_detector: EmotionDetector,
        get_current_identity: Callable[[], Optional[str]],
        face_recognition_system=None,
        get_camera: Optional[Callable] = None,
        set_expression: Optional[Callable[[str, float], None]] = None,
        timeout: Optional[float] = None
    ):
        """Initialize the Tool Handler.

        Args:
            strands_agent: The Strands Agent instance for memory and web search.
            emotion_detector: The Emotion Detector instance for emotion queries.
            get_current_identity: Callable that returns the currently recognized
                person's name, or None if no one is identified.
            face_recognition_system: Optional FaceRecognitionSystem for enrollment.
            get_camera: Optional callable that returns the camera source for enrollment.
            timeout: Optional override for tool execution timeout in seconds.
                Defaults to DEFAULT_TIMEOUT (10 seconds).
        """
        self.strands_agent = strands_agent
        self.emotion_detector = emotion_detector
        self.get_current_identity = get_current_identity
        self.face_recognition_system = face_recognition_system
        self.get_camera = get_camera
        self.set_expression = set_expression
        self._timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="ToolHandler"
        )

    def get_tool_config_json(self) -> str:
        """Return tool configuration JSON for Nova Sonic 2 promptStart event.

        Returns:
            JSON string containing the toolUse configuration with all
            registered tool definitions.
        """
        config = {
            "toolUse": {
                "tools": self.TOOL_DEFINITIONS
            }
        }
        return json.dumps(config)

    def build_prompt_start_tools_section(self) -> str:
        """Build the toolUse section for the promptStart event JSON.

        Returns:
            JSON string of just the toolUse section (without outer wrapper),
            suitable for embedding in a promptStart event payload.
        """
        tools_section = {
            "tools": self.TOOL_DEFINITIONS
        }
        return json.dumps(tools_section)

    async def handle_tool_use(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool call from Nova Sonic 2.

        Routes the tool call to the appropriate handler based on tool_name:
        - "recall_memory" → Strands Agent memory recall
        - "search_web" → Strands Agent web search
        - "get_emotion" → Emotion Detector direct read

        Args:
            tool_name: Name of the tool to execute.
            parameters: Tool parameters from the Nova Sonic 2 toolUse event.

        Returns:
            Result dictionary with tool output on success, or error dictionary
            with 'error' and 'message' keys on failure/timeout.
        """
        if not tool_name:
            return {"error": "invalid_params", "message": "Tool name is required"}

        print(f"🔧 Tool called: {tool_name} with params: {parameters}")

        # Route to appropriate handler
        if tool_name == "recall_memory":
            return await self._handle_recall_memory(parameters)
        elif tool_name == "get_emotion":
            return await self._handle_get_emotion(parameters)
        elif tool_name == "set_expression":
            return await self._handle_set_expression(parameters)
        elif tool_name == "search_web":
            return await self._handle_search_web(parameters)
        elif tool_name == "enroll_face":
            return await self._handle_enroll_face(parameters)
        elif tool_name == "forget_person":
            return await self._handle_forget_person(parameters)
        else:
            return {
                "error": "unknown_tool",
                "message": f"Tool '{tool_name}' not found"
            }

    async def _handle_recall_memory(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle recall_memory tool invocation via Strands Agent.

        Scopes the memory query to the currently identified person.
        """
        person_name = self.get_current_identity()
        if not person_name:
            return {
                "result": "No person is currently identified. Cannot recall memories."
            }

        query = parameters.get("query")

        try:
            result = await self._run_with_timeout(
                self.strands_agent.recall_memory, person_name, query
            )
            if result:
                return {"result": result}
            else:
                return {"result": f"No memories found for {person_name}."}
        except TimeoutError:
            return {
                "error": "timeout",
                "message": "Memory recall timed out"
            }
        except Exception as e:
            logger.error(f"recall_memory failed: {e}")
            return {
                "error": "execution_failed",
                "message": f"Memory recall failed: {str(e)}"
            }

    async def _handle_get_emotion(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle get_emotion tool invocation via Emotion Detector.

        Reads directly from the Emotion Detector's current state.
        """
        try:
            state = self.emotion_detector.get_current_state()
            return {
                "result": {
                    "emotion": str(state["emotion"]),
                    "confidence": float(state["confidence"])
                }
            }
        except Exception as e:
            logger.error(f"get_emotion failed: {e}")
            return {
                "error": "execution_failed",
                "message": f"Emotion detection failed: {str(e)}"
            }

    # Map assistant-facing emotion words (and common synonyms) to the avatar's
    # available VRM expressions: happy, angry, sad, relaxed, neutral.
    _EXPRESSION_ALIASES = {
        "happy": "happy", "joy": "happy", "joyful": "happy", "excited": "happy",
        "glad": "happy", "smile": "happy", "cheerful": "happy",
        "sad": "sad", "unhappy": "sad", "upset": "sad", "down": "sad",
        "disappointed": "sad", "sorrow": "sad",
        "angry": "angry", "anger": "angry", "mad": "angry", "frustrated": "angry",
        "relaxed": "relaxed", "calm": "relaxed", "content": "relaxed",
        "peaceful": "relaxed", "reassuring": "relaxed",
        "neutral": "neutral", "resting": "neutral", "none": "neutral",
    }

    async def _handle_set_expression(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle set_expression: set the avatar's facial emotion in the UI."""
        raw = str(parameters.get("emotion", "")).strip().lower()
        expression = self._EXPRESSION_ALIASES.get(raw)
        if expression is None:
            return {
                "error": "invalid_params",
                "message": (
                    f"Unknown emotion '{raw}'. Use one of: happy, sad, angry, "
                    "relaxed, neutral."
                ),
            }

        try:
            intensity = float(parameters.get("intensity", 1.0))
        except (TypeError, ValueError):
            intensity = 1.0
        intensity = max(0.0, min(1.0, intensity))
        # Neutral clears the expression.
        weight = 0.0 if expression == "neutral" else intensity

        if self.set_expression is None:
            return {
                "error": "unavailable",
                "message": "Expression control is not available",
            }

        try:
            self.set_expression(expression, weight)
            return {"result": {"expression": expression, "intensity": weight}}
        except Exception as e:
            logger.error(f"set_expression failed: {e}")
            return {
                "error": "execution_failed",
                "message": f"Could not set expression: {str(e)}",
            }

    async def _handle_search_web(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle search_web tool invocation via Strands Agent.

        Validates that a query parameter is provided.
        """
        query = parameters.get("query")
        if not query or not str(query).strip():
            return {
                "error": "invalid_params",
                "message": "The 'query' parameter is required for search_web"
            }

        try:
            result = await self._run_with_timeout(
                self.strands_agent.search_web, str(query).strip()
            )
            return {"result": result}
        except TimeoutError:
            return {
                "error": "timeout",
                "message": "Web search timed out"
            }
        except Exception as e:
            logger.error(f"search_web failed: {e}")
            return {
                "error": "execution_failed",
                "message": f"Web search failed: {str(e)}"
            }

    async def _handle_enroll_face(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle enroll_face tool invocation.

        Enrolls the face currently visible to the camera under the given name.
        """
        name = parameters.get("name", "").strip()
        print(f"📸 enroll_face tool called with name='{name}'")
        
        if not name:
            return {
                "error": "invalid_params",
                "message": "A name is required to enroll a face"
            }

        if not self.face_recognition_system:
            print("   ❌ Face recognition system not initialized")
            return {
                "error": "not_available",
                "message": "Face recognition system is not initialized"
            }

        if not self.get_camera:
            print("   ❌ No get_camera callable")
            return {
                "error": "not_available",
                "message": "Camera is not available for enrollment"
            }

        camera = self.get_camera()
        if camera is None:
            print("   ❌ Camera is None")
            return {
                "error": "not_available",
                "message": "Camera is not currently active"
            }

        # Wrap camera to use latest frame from face tracker (avoids contention)
        class FrameProxy:
            """Provides .read() that returns the latest frame from face tracker."""
            def __init__(self, cam):
                self._cam = cam
                self._read_count = 0
            def read(self):
                self._read_count += 1
                # Use the cached latest frame if available (set by face tracker loop)
                if hasattr(self._cam, '_latest_frame') and self._cam._latest_frame is not None:
                    frame = self._cam._latest_frame.copy()
                    if self._read_count <= 3:
                        print(f"   📷 FrameProxy: got frame {frame.shape}")
                    return True, frame
                if self._read_count <= 3:
                    print(f"   📷 FrameProxy: no frame available (has attr: {hasattr(self._cam, '_latest_frame')}, is None: {getattr(self._cam, '_latest_frame', 'missing') is None})")
                # Fallback: try reading from camera directly
                if hasattr(self._cam, 'camera') and self._cam.camera is not None:
                    return self._cam.camera.read()
                return False, None

        # If we got a face_tracker object (has _latest_frame), use the proxy
        # Otherwise use the camera directly
        if hasattr(camera, '_latest_frame'):
            enrollment_source = FrameProxy(camera)
        else:
            enrollment_source = camera

        try:
            print(f"   📷 Starting enrollment for '{name}'...")
            # Use longer timeout for enrollment (camera sharing + multiple captures)
            import functools
            loop = asyncio.get_running_loop()
            call = functools.partial(
                self.face_recognition_system.enroll, name, enrollment_source, 3, 10.0
            )
            success, message = await asyncio.wait_for(
                loop.run_in_executor(self._executor, call),
                timeout=15.0  # Slightly longer than internal timeout
            )
            print(f"   {'✅' if success else '❌'} Enrollment result: {message}")
            if success:
                return {"result": f"Successfully enrolled {name}. I'll recognize them next time!"}
            else:
                return {"result": f"Enrollment failed: {message}"}
        except TimeoutError:
            print("   ❌ Enrollment timed out")
            return {
                "error": "timeout",
                "message": "Face enrollment timed out"
            }
        except Exception as e:
            print(f"   ❌ Enrollment exception: {e}")
            logger.error(f"enroll_face failed: {e}")
            return {
                "error": "execution_failed",
                "message": f"Enrollment failed: {str(e)}"
            }

    async def _handle_forget_person(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle forget_person tool invocation.

        Removes face enrollment and clears stored memories for the person.
        """
        name = parameters.get("name", "").strip()
        print(f"🗑️ forget_person tool called with name='{name}'")

        if not name:
            return {
                "error": "invalid_params",
                "message": "A name is required to forget a person"
            }

        results = []

        # Remove face enrollment
        if self.face_recognition_system:
            removed = self.face_recognition_system.remove_person(name)
            if removed:
                results.append(f"Removed face enrollment for '{name}'")
                print(f"   ✅ Face enrollment removed for '{name}'")
            else:
                results.append(f"No face enrollment found for '{name}'")
                print(f"   ⚠️ No face enrollment found for '{name}'")

        # Note: AgentCore Memory doesn't have a delete API exposed here,
        # but the face removal means they won't be recognized anymore
        # and new conversations won't be stored under their name.
        results.append(f"'{name}' will no longer be recognized by the camera")

        return {"result": ". ".join(results)}

    async def _run_with_timeout(self, func, *args) -> Any:
        """Run a synchronous function with timeout enforcement.

        Executes the function in a thread pool and enforces the configured
        timeout. Raises TimeoutError if the function exceeds the timeout.

        Args:
            func: The synchronous function to execute.
            *args: Arguments to pass to the function.

        Returns:
            The function's return value.

        Raises:
            TimeoutError: If execution exceeds the configured timeout.
        """
        import functools

        loop = asyncio.get_running_loop()
        call = functools.partial(func, *args)
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, call),
                timeout=self._timeout
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Tool execution exceeded {self._timeout}s timeout"
            )

    def shutdown(self) -> None:
        """Clean up resources used by the tool handler."""
        self._executor.shutdown(wait=False)
        logger.info("Tool Handler shut down")
