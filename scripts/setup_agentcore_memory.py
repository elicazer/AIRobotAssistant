#!/usr/bin/env python3
"""
One-time setup script for Amazon Bedrock AgentCore Memory resource.

Creates a Memory resource with three LTM (Long-Term Memory) strategies:
  - session_summaries: Condensed records of past conversations
  - user_preferences: Likes, dislikes, stated preferences
  - semantic_facts: Factual information learned about the person

Usage:
    python scripts/setup_agentcore_memory.py [--region REGION]

Prerequisites:
    - AWS credentials configured (via environment, ~/.aws/credentials, or IAM role)
    - Appropriate IAM permissions for Bedrock AgentCore Memory operations

Outputs the memory_resource_id to be added to config/voice_assistant_settings.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_REGION = "us-east-1"
SETTINGS_PATH = Path("config/voice_assistant_settings.json")

# LTM strategy definitions
LTM_STRATEGIES = [
    {
        "name": "session_summaries",
        "description": "Condensed records of past conversations with the person",
    },
    {
        "name": "user_preferences",
        "description": "Likes, dislikes, and stated preferences of the person",
    },
    {
        "name": "semantic_facts",
        "description": "Factual information learned about the person",
    },
]


def create_memory_resource(region: str = DEFAULT_REGION) -> str:
    """Create an AgentCore Memory resource with three LTM strategies.

    Uses existing AWS credentials to create the memory resource in the
    specified region. Configures session_summaries, user_preferences,
    and semantic_facts strategies.

    Args:
        region: AWS region to create the resource in.

    Returns:
        The memory resource ID string.

    Raises:
        RuntimeError: If resource creation fails.
    """
    try:
        import boto3
    except ImportError:
        raise RuntimeError(
            "boto3 is required but not installed. "
            "Install it with: pip install boto3"
        )

    logger.info(f"Creating AgentCore Memory resource in region '{region}'...")

    try:
        # AgentCore Memory uses the 'bedrock-agentcore-control' client for
        # control plane operations (create/get/delete memory resources)
        control_client = boto3.client("bedrock-agentcore-control", region_name=region)

        # Create the memory resource with all three LTM strategies
        response = control_client.create_memory(
            name="inmoov_robot_personalization",
            description=(
                "Long-term memory for InMoov robot voice assistant. "
                "Stores per-person conversation history, preferences, and facts."
            ),
            eventExpiryDuration=90,  # Keep raw events for 90 days
            memoryStrategies=[
                {
                    "summaryMemoryStrategy": {
                        "name": "SessionSummarizer",
                        "namespaceTemplates": ["/summaries/{actorId}/{sessionId}/"]
                    }
                },
                {
                    "userPreferenceMemoryStrategy": {
                        "name": "UserPreferenceExtractor",
                        "namespaceTemplates": ["/users/{actorId}/preferences/"]
                    }
                },
                {
                    "semanticMemoryStrategy": {
                        "name": "SemanticFactExtractor",
                        "namespaceTemplates": ["/users/{actorId}/facts/"]
                    }
                }
            ],
        )

        memory_id = response.get("memory", {}).get("id")
        if not memory_id:
            raise RuntimeError(
                "Memory resource created but no memory ID returned in response. "
                f"Full response: {json.dumps(response, default=str)}"
            )

        logger.info(f"Memory resource created with ID: {memory_id}")

        # Poll until the memory becomes ACTIVE
        import time as _time
        logger.info("Waiting for memory resource to become ACTIVE...")
        for _ in range(30):  # Wait up to 5 minutes
            status_response = control_client.get_memory(memoryId=memory_id)
            status = status_response.get("memory", {}).get("status")
            if status == "ACTIVE":
                logger.info("Memory resource is now ACTIVE.")
                return memory_id
            elif status == "FAILED":
                raise RuntimeError("Memory resource creation FAILED.")
            _time.sleep(10)

        raise RuntimeError("Timed out waiting for memory resource to become ACTIVE.")

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to create memory resource: {e}")


def update_settings_file(memory_resource_id: str) -> bool:
    """Update the voice assistant settings file with the new resource ID.

    Args:
        memory_resource_id: The memory resource ID to write into settings.

    Returns:
        True if settings were updated, False if user declined or file not found.
    """
    if not SETTINGS_PATH.exists():
        logger.warning(f"Settings file not found at {SETTINGS_PATH}")
        return False

    try:
        settings = json.loads(SETTINGS_PATH.read_text())
        current_id = settings.get("agentcore_memory_resource_id", "")

        if current_id and current_id != memory_resource_id:
            logger.warning(
                f"Settings already has a memory resource ID: {current_id}"
            )
            response = input("Overwrite existing resource ID? [y/N]: ").strip().lower()
            if response != "y":
                logger.info("Skipping settings update.")
                return False

        settings["agentcore_memory_resource_id"] = memory_resource_id
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n")
        logger.info(f"Updated {SETTINGS_PATH} with memory resource ID.")
        return True

    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to update settings file: {e}")
        return False


def main():
    """Run the AgentCore Memory setup.

    Parses command-line arguments, creates the memory resource,
    and prints configuration instructions.
    """
    parser = argparse.ArgumentParser(
        description="Set up Amazon Bedrock AgentCore Memory for InMoov robot"
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region for the memory resource (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--no-update-settings",
        action="store_true",
        help="Don't automatically update the settings file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  InMoov Robot — AgentCore Memory Setup")
    print("=" * 60)
    print()

    try:
        memory_resource_id = create_memory_resource(region=args.region)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    print()
    print("-" * 60)
    print("  Setup Complete!")
    print("-" * 60)
    print()
    print(f"  Memory Resource ID: {memory_resource_id}")
    print(f"  Region:             {args.region}")
    print(f"  LTM Strategies:     {', '.join(s['name'] for s in LTM_STRATEGIES)}")
    print()

    # Attempt to update settings file automatically
    if not args.no_update_settings:
        updated = update_settings_file(memory_resource_id)
        if not updated:
            print("  To configure manually, add this to your settings file:")
            print(f"    {SETTINGS_PATH}")
            print()
            print(f'    "agentcore_memory_resource_id": "{memory_resource_id}"')
    else:
        print("  Add this to your settings file:")
        print(f"    {SETTINGS_PATH}")
        print()
        print(f'    "agentcore_memory_resource_id": "{memory_resource_id}"')

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
