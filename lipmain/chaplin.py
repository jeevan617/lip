import cv2
import time
from ollama import AsyncClient
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import os
import asyncio
from threading import Thread


class ChaplinOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


class Chaplin:
    def __init__(self):
        self.vsr_model = None

        # thread stuff
        self.executor = ThreadPoolExecutor(max_workers=1)

        # setup async ollama client
        self.ollama_client = AsyncClient()

        # setup asyncio event loop in background thread
        self.loop = asyncio.new_event_loop()
        self.async_thread = ThreadPoolExecutor(max_workers=1)
        self.async_thread.submit(self._run_event_loop)

        # sequence tracking to ensure outputs are typed in order
        self.next_sequence_to_type = 0
        self.current_sequence = 0  # counter for assigning sequence numbers
        self.typing_lock = None  # will be created in async loop
        self._init_async_resources()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _init_async_resources(self):
        """Initialize async resources in the async loop"""
        future = asyncio.run_coroutine_threadsafe(
            self._create_async_lock(), self.loop)
        future.result()  # wait for it to complete

    async def _create_async_lock(self):
        """Create asyncio.Lock and Condition in the event loop's context"""
        self.typing_lock = asyncio.Lock()
        self.typing_condition = asyncio.Condition(self.typing_lock)

    async def correct_output_async(self, output, sequence_num):
        # perform inference on the raw output to get back a "correct" version
        response = await self.ollama_client.chat(
            model='qwen3:4b',
            messages=[
                {
                    'role': 'system',
                    'content': f"You are an assistant that helps make corrections to the output of a lipreading model. The text you will receive was transcribed using a video-to-text system that attempts to lipread the subject speaking in the video, so the text will likely be imperfect. The input text will also be in all-caps, although your respose should be capitalized correctly and should NOT be in all-caps.\n\nIf something seems unusual, assume it was mistranscribed. Do your best to infer the words actually spoken, and make changes to the mistranscriptions in your response. Do not add more words or content, just change the ones that seem to be out of place (and, therefore, mistranscribed). Do not change even the wording of sentences, just individual words that look nonsensical in the context of all of the other words in the sentence.\n\nAlso, add correct punctuation to the entire text. ALWAYS end each sentence with the appropriate sentence ending: '.', '?', or '!'. \n\nReturn the corrected text in the format of 'list_of_changes' and 'corrected_text'."
                },
                {
                    'role': 'user',
                    'content': f"Transcription:\n\n{output}"
                }
            ],
            format=ChaplinOutput.model_json_schema()
        )

        # get only the corrected text
        chat_output = ChaplinOutput.model_validate_json(
            response['message']['content'])

        # if last character isn't a sentence ending (happens sometimes), add a period
        chat_output.corrected_text = chat_output.corrected_text.strip()
        if chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'

        # add space at the end
        chat_output.corrected_text += ' '

        return chat_output.corrected_text

    def perform_inference(self, video_path):
        # perform inference on the video with the vsr model
        print("Running VSR model...")
        output = self.vsr_model(video_path)

        # print the raw output to console
        print(f"\n\033[48;5;21m\033[97m\033[1m RAW OUTPUT \033[0m: {output}\n")

        # assign sequence number for this task
        sequence_num = self.current_sequence
        self.current_sequence += 1

        # start the async LLM correction (non-blocking) with sequence number
        corrected_text_future = asyncio.run_coroutine_threadsafe(
            self.correct_output_async(output, sequence_num),
            self.loop
        )

        return {
            "output": output,
            "corrected_text_future": corrected_text_future
        }

    def process_video_frame(self, frame):
        # This is a placeholder for the actual video processing logic
        # In a real application, you would perform inference on the frame here
        return "Processed frame"
