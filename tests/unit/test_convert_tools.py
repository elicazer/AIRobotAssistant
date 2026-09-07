"""Unit tests for convert_tools_nova_to_openai function."""

import json
import pytest

from src.openai_realtime_client import convert_tools_nova_to_openai


class TestConvertToolsNovaToOpenai:
    """Tests for converting Nova Sonic tool config to OpenAI function format."""

    def test_single_tool_full_spec(self):
        """Convert a single tool with all fields present."""
        config = json.dumps({
            "toolUse": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "recall_memory",
                            "description": "Recall a memory by query",
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        })

        result = convert_tools_nova_to_openai(config)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "recall_memory"
        assert result[0]["description"] == "Recall a memory by query"
        assert result[0]["parameters"] == {
            "type": "object",
            "properties": {"query": {"type": "string"}}
        }

    def test_multiple_tools(self):
        """Convert multiple tools in a single config."""
        config = json.dumps({
            "toolUse": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "recall_memory",
                            "description": "Recall memory",
                            "inputSchema": {"json": {"type": "object"}}
                        }
                    },
                    {
                        "toolSpec": {
                            "name": "get_emotion",
                            "description": "Get current emotion",
                            "inputSchema": {"json": {"type": "object", "properties": {}}}
                        }
                    }
                ]
            }
        })

        result = convert_tools_nova_to_openai(config)

        assert len(result) == 2
        assert result[0]["name"] == "recall_memory"
        assert result[1]["name"] == "get_emotion"
        assert all(t["type"] == "function" for t in result)

    def test_missing_description_uses_empty_string(self):
        """Missing description field defaults to empty string."""
        config = json.dumps({
            "toolUse": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "search_web",
                            "inputSchema": {"json": {"type": "object"}}
                        }
                    }
                ]
            }
        })

        result = convert_tools_nova_to_openai(config)

        assert len(result) == 1
        assert result[0]["description"] == ""

    def test_missing_input_schema_uses_empty_dict(self):
        """Missing inputSchema field defaults to empty dict for parameters."""
        config = json.dumps({
            "toolUse": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "get_emotion",
                            "description": "Detect emotion"
                        }
                    }
                ]
            }
        })

        result = convert_tools_nova_to_openai(config)

        assert len(result) == 1
        assert result[0]["parameters"] == {}

    def test_missing_input_schema_json_uses_empty_dict(self):
        """inputSchema present but missing 'json' key defaults to empty dict."""
        config = json.dumps({
            "toolUse": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "enroll_face",
                            "description": "Enroll a face",
                            "inputSchema": {}
                        }
                    }
                ]
            }
        })

        result = convert_tools_nova_to_openai(config)

        assert len(result) == 1
        assert result[0]["parameters"] == {}

    def test_invalid_json_returns_empty_list(self):
        """Invalid JSON input returns empty list."""
        result = convert_tools_nova_to_openai("not valid json {{{")
        assert result == []

    def test_empty_string_returns_empty_list(self):
        """Empty string input returns empty list."""
        result = convert_tools_nova_to_openai("")
        assert result == []

    def test_none_input_returns_empty_list(self):
        """None input returns empty list."""
        result = convert_tools_nova_to_openai(None)
        assert result == []

    def test_empty_tools_list_returns_empty_list(self):
        """Empty tools array returns empty list."""
        config = json.dumps({"toolUse": {"tools": []}})
        result = convert_tools_nova_to_openai(config)
        assert result == []

    def test_missing_tool_use_key_returns_empty_list(self):
        """Missing 'toolUse' key returns empty list."""
        config = json.dumps({"other": "data"})
        result = convert_tools_nova_to_openai(config)
        assert result == []

    def test_missing_tools_key_returns_empty_list(self):
        """Missing 'tools' key inside toolUse returns empty list."""
        config = json.dumps({"toolUse": {"other": "data"}})
        result = convert_tools_nova_to_openai(config)
        assert result == []

    def test_tool_without_tool_spec_is_skipped(self):
        """Tools missing toolSpec are skipped, others still converted."""
        config = json.dumps({
            "toolUse": {
                "tools": [
                    {"noSpec": True},
                    {
                        "toolSpec": {
                            "name": "valid_tool",
                            "description": "A valid tool",
                            "inputSchema": {"json": {"type": "object"}}
                        }
                    }
                ]
            }
        })

        result = convert_tools_nova_to_openai(config)

        assert len(result) == 1
        assert result[0]["name"] == "valid_tool"

    def test_tool_without_name_is_skipped(self):
        """Tools with toolSpec but missing name are skipped."""
        config = json.dumps({
            "toolUse": {
                "tools": [
                    {
                        "toolSpec": {
                            "description": "No name tool",
                            "inputSchema": {"json": {"type": "object"}}
                        }
                    }
                ]
            }
        })

        result = convert_tools_nova_to_openai(config)
        assert result == []
