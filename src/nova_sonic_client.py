"""
Nova Sonic Client for Voice Interaction
Handles speech-to-speech with AWS Bedrock Nova Sonic
"""

import os
import asyncio
import base64
import json
import uuid
import pyaudio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

try:
    from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
    from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
    from aws_sdk_bedrock_runtime.config import Config
    from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    print("⚠️  Nova Sonic SDK not available")
    print("Install the aws-sdk-bedrock-runtime package")

# Audio configuration
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024


class NovaSonicClient:
    """
    Client for Amazon Nova Sonic speech-to-speech interaction
    
    Callbacks:
        on_user_text: Called when user speech is transcribed
        on_assistant_text: Called when assistant response text is available
        on_audio_output: Called when audio chunk is ready for playback
    """
    
    def __init__(
        self,
        model_id: str = 'amazon.nova-sonic-v1:0',
        region: str = 'us-east-1',
        voice_id: str = 'matthew',
        system_prompt: Optional[str] = None,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None
    ):
        if not BEDROCK_AVAILABLE:
            raise ImportError("AWS Bedrock SDK not available. Install requirements_voice.txt")
        
        self.model_id = model_id
        self.region = region
        self.voice_id = voice_id
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.client = None
        self.stream = None
        self.response = None
        self.is_active = False
        
        # Unique identifiers for this session
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        
        # Queues and state
        self.audio_queue = asyncio.Queue()
        self.role = None
        self.display_assistant_text = False
        
        # System prompt
        self.system_prompt = system_prompt or (
            "You are a friendly robot assistant. Keep your responses short and natural, "
            "generally two or three sentences. You are speaking out loud, so be conversational."
        )
        
        # Callbacks
        self.on_user_text: Optional[Callable[[str], None]] = None
        self.on_assistant_text: Optional[Callable[[str], None]] = None
        self.on_audio_output: Optional[Callable[[bytes], None]] = None
        self.on_audio_chunk: Optional[Callable[[bytes], None]] = None  # Real-time audio processing
        
        # Thread pool for non-blocking callbacks
        self.callback_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio_callback")
    
    def _safe_callback(self, audio_data):
        """Safely execute audio chunk callback without blocking audio"""
        try:
            if self.on_audio_chunk:
                self.on_audio_chunk(audio_data)
        except Exception as e:
            # Silently ignore callback errors to prevent audio disruption
            pass
    
    def _initialize_client(self):
        """Initialize the Bedrock client"""
        print(f"Connecting to Bedrock in {self.region}...")
        
        # Get credentials from boto3 session (reads from ~/.aws/credentials)
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if not credentials:
            raise Exception("AWS credentials not found. Run 'aws configure' or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY")
        
        # Set environment variables for Smithy SDK
        import os
        os.environ['AWS_ACCESS_KEY_ID'] = credentials.access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = credentials.secret_key
        if credentials.token:
            os.environ['AWS_SESSION_TOKEN'] = credentials.token
        
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.client = BedrockRuntimeClient(config=config)
        print("Bedrock client initialized")
    
    async def send_event(self, event_json: str):
        """Send an event to the stream"""
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        await self.stream.input_stream.send(event)
    
    async def start_session(self):
        """Start a new session with Nova Sonic"""
        if not self.client:
            self._initialize_client()
        
        print("Starting Nova Sonic session...")
        
        # Initialize the stream
        print(f"Opening bidirectional stream for model: {self.model_id}")
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        print("Stream opened successfully")
        self.is_active = True
        
        # Send session start event
        session_start = '''
        {
          "event": {
            "sessionStart": {
              "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.7
              }
            }
          }
        }
        '''
        await self.send_event(session_start)
        
        # Send prompt start event
        prompt_start = f'''
        {{
          "event": {{
            "promptStart": {{
              "promptName": "{self.prompt_name}",
              "textOutputConfiguration": {{
                "mediaType": "text/plain"
              }},
              "audioOutputConfiguration": {{
                "mediaType": "audio/lpcm",
                "sampleRateHertz": {OUTPUT_SAMPLE_RATE},
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": "{self.voice_id}",
                "encoding": "base64",
                "audioType": "SPEECH"
              }}
            }}
          }}
        }}
        '''
        await self.send_event(prompt_start)
        
        # Send system prompt
        text_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}",
                    "type": "TEXT",
                    "interactive": false,
                    "role": "SYSTEM",
                    "textInputConfiguration": {{
                        "mediaType": "text/plain"
                    }}
                }}
            }}
        }}
        '''
        await self.send_event(text_content_start)
        
        text_input = f'''
        {{
            "event": {{
                "textInput": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}",
                    "content": "{self.system_prompt}"
                }}
            }}
        }}
        '''
        await self.send_event(text_input)
        
        text_content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}"
                }}
            }}
        }}
        '''
        await self.send_event(text_content_end)
        
        # Start processing responses
        self.response = asyncio.create_task(self._process_responses())
        
        print("Nova Sonic session started!")
    
    async def start_audio_input(self):
        """Start audio input stream"""
        audio_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}",
                    "type": "AUDIO",
                    "interactive": true,
                    "role": "USER",
                    "audioInputConfiguration": {{
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": {INPUT_SAMPLE_RATE},
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }}
                }}
            }}
        }}
        '''
        await self.send_event(audio_content_start)
    
    async def send_audio_chunk(self, audio_bytes: bytes):
        """Send an audio chunk to the stream"""
        if not self.is_active:
            return
        
        blob = base64.b64encode(audio_bytes)
        audio_event = f'''
        {{
            "event": {{
                "audioInput": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}",
                    "content": "{blob.decode('utf-8')}"
                }}
            }}
        }}
        '''
        await self.send_event(audio_event)
    
    async def end_audio_input(self):
        """End audio input stream"""
        audio_content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}"
                }}
            }}
        }}
        '''
        await self.send_event(audio_content_end)
    
    async def end_session(self):
        """End the session"""
        if not self.is_active:
            return
        
        print("\nEnding Nova Sonic session...")
        
        try:
            prompt_end = f'''
            {{
                "event": {{
                    "promptEnd": {{
                        "promptName": "{self.prompt_name}"
                    }}
                }}
            }}
            '''
            await self.send_event(prompt_end)
            
            session_end = '''
            {
                "event": {
                    "sessionEnd": {}
                }
            }
            '''
            await self.send_event(session_end)
            
            # Close the stream
            await self.stream.input_stream.close()
        except Exception:
            # Suppress errors during cleanup - stream may already be closed
            pass
        finally:
            self.is_active = False
    
    async def _process_responses(self):
        """Process responses from the stream"""
        try:
            while self.is_active:
                output = await self.stream.await_output()
                result = await output[1].receive()
                
                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode('utf-8')
                    json_data = json.loads(response_data)
                    
                    if 'event' in json_data:
                        # Handle content start event
                        if 'contentStart' in json_data['event']:
                            content_start = json_data['event']['contentStart']
                            self.role = content_start['role']
                            
                            # Check for speculative content
                            if 'additionalModelFields' in content_start:
                                additional_fields = json.loads(content_start['additionalModelFields'])
                                if additional_fields.get('generationStage') == 'SPECULATIVE':
                                    self.display_assistant_text = True
                                else:
                                    self.display_assistant_text = False
                        
                        # Handle text output event
                        elif 'textOutput' in json_data['event']:
                            text = json_data['event']['textOutput']['content']
                            
                            if self.role == "ASSISTANT" and self.display_assistant_text:
                                print(f"Assistant: {text}")
                                if self.on_assistant_text:
                                    self.on_assistant_text(text)
                            elif self.role == "USER":
                                print(f"User: {text}")
                                if self.on_user_text:
                                    self.on_user_text(text)
                        
                        # Handle audio output
                        elif 'audioOutput' in json_data['event']:
                            audio_content = json_data['event']['audioOutput']['content']
                            audio_bytes = base64.b64decode(audio_content)
                            await self.audio_queue.put(audio_bytes)
                            
                            if self.on_audio_output:
                                self.on_audio_output(audio_bytes)
        
        except asyncio.CancelledError:
            # Normal cancellation during shutdown - suppress
            pass
        except Exception as e:
            # Only log unexpected errors
            if "InvalidStateError" not in str(e) and "CANCELLED" not in str(e):
                print(f"Error processing responses: {e}")
                import traceback
                traceback.print_exc()
    
    async def play_audio(self):
        """Play audio responses"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
            output_device_index=self.output_device_index
        )
        
        print("Audio playback ready...")
        
        try:
            while self.is_active:
                audio_data = await self.audio_queue.get()
                
                # Play audio immediately (don't block on callback)
                stream.write(audio_data)
                
                # Call audio chunk callback in separate thread (non-blocking)
                if self.on_audio_chunk:
                    self.callback_executor.submit(self._safe_callback, audio_data)
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Audio playback stopped.")
    
    async def capture_audio(self):
        """Capture audio from microphone and send to Nova Sonic"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=self.input_device_index
        )
        
        print("🎤 Listening... Speak into your microphone!")
        print("Press Ctrl+C to stop\n")
        
        await self.start_audio_input()
        
        try:
            while self.is_active:
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                await self.send_audio_chunk(audio_data)
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error capturing audio: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Audio capture stopped.")
            await self.end_audio_input()
