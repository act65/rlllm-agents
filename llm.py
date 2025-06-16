import base64
import os
from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


def generate(model_name, prompt, temperature):
    client = genai.Client(
        # api_key=os.environ.get("GEMINI_API_KEY"),
        api_key=GEMINI_API_KEY,
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="text/plain",
    )

    output = ""
    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text is not None:
            output += chunk.text

    print('output', output)
    return output

if __name__ == "__main__":
    print(generate(
        'gemini-2.5-flash-preview-04-17',
        "What is 3 x 432?",
        0.1)
    )