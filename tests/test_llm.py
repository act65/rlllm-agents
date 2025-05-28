import pytest
from unittest.mock import patch, MagicMock
from llm_explorer.llm import generate # Assuming llm.py is in src/llm_explorer

# Test for the generate function
@patch('llm_explorer.llm.os.environ.get')
@patch('llm_explorer.llm.genai.Client')
def test_generate_successful_call(mock_genai_client, mock_os_environ_get):
    # --- Setup Mocks ---
    # Mock environment variable
    mock_os_environ_get.return_value = "test_api_key"

    # Mock the genai client and its stream method
    mock_client_instance = MagicMock()
    mock_genai_client.return_value = mock_client_instance

    # Mock the response from generate_content_stream
    # Each item in the list is a chunk with a 'text' attribute
    mock_stream_response = [
        MagicMock(text="Hello "),
        MagicMock(text="World"),
        MagicMock(text="!"),
    ]
    mock_client_instance.models.generate_content_stream.return_value = mock_stream_response

    # --- Call the function under test ---
    model_name = "gemini-test-model"
    prompt = "Test prompt"
    temperature = 0.5
    result = generate(model_name, prompt, temperature)

    # --- Assertions ---
    # Check that the API key was fetched
    mock_os_environ_get.assert_called_once_with("GEMINI_API_KEY")

    # Check that the client was initialized
    mock_genai_client.assert_called_once_with(api_key="test_api_key")

    # Check that generate_content_stream was called with correct parameters
    # We need to access the call_args from the mock to check the 'contents' and 'config'
    # This can be a bit verbose due to the structure of 'types.Content' and 'types.GenerateContentConfig'
    args, kwargs = mock_client_instance.models.generate_content_stream.call_args
    
    assert kwargs['model'] == model_name
    
    # Check contents (simplified check for the prompt text)
    called_contents = kwargs['contents']
    assert len(called_contents) == 1
    assert called_contents[0].role == "user"
    assert len(called_contents[0].parts) == 1
    # To access Part.from_text(text=prompt), we would need to know how types.Part stores the text
    # Assuming it has a 'text' attribute or similar after creation.
    # For simplicity, we'll trust it was constructed correctly if the prompt is in there somewhere.
    # A more robust check would involve inspecting the Part object structure.
    # Let's assume the prompt makes it into one of the Part's attributes.
    # This part of the assertion might need adjustment based on actual Part structure.
    assert prompt in str(called_contents[0].parts[0]) # A bit of a loose check


    # Check config
    called_config = kwargs['config']
    assert called_config.temperature == temperature
    assert called_config.response_mime_type == "text/plain"

    # Check that the result is the concatenation of text parts from the stream
    assert result == "Hello World!"

@patch('llm_explorer.llm.os.environ.get')
@patch('llm_explorer.llm.genai.Client')
def test_generate_empty_key(mock_genai_client, mock_os_environ_get):
    # Test what happens if API key is not found
    mock_os_environ_get.return_value = None
    
    # We expect the genai.Client to be called with api_key=None
    # and then it's up to the google.genai library to handle this
    # (e.g., raise an error or try other auth methods).
    # For this unit test, we just ensure our code passes None to it.
    
    # Mock the client to prevent actual calls and to avoid errors during client init
    mock_client_instance = MagicMock()
    mock_genai_client.return_value = mock_client_instance
    mock_stream_response = [MagicMock(text="test")] # Dummy response
    mock_client_instance.models.generate_content_stream.return_value = mock_stream_response

    generate("model", "prompt", 0.5)
    
    mock_genai_client.assert_called_once_with(api_key=None)
