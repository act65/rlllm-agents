import base64
import os
from google import genai
from google.genai import types

def generate(model_name, prompt, temperature):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
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
        output += chunk.text

    return output

if __name__ == "__main__":
    print(generate(
        'gemini-2.5-flash-preview-04-17',
        "What is 3 x 432?",
        0.1)
    )