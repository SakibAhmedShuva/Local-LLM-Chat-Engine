# gguf_handler.py
import os
import logging
from llama_cpp import Llama, LlamaGrammar
from llama_index.core.llms import MessageRole # Using for message structure consistency

# Constants for GGUF specific parameters
DEFAULT_N_GPU_LAYERS = -1 # -1 to offload all possible layers to GPU
DEFAULT_N_CTX = 4096      # Default context window for loading
DEFAULT_MAX_TOKENS = 512  # Default max tokens for generation
DEFAULT_GGUF_TEMPERATURE = 0.7
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.95
DEFAULT_REPEAT_PENALTY = 1.1

logger = logging.getLogger(__name__)

if not os.environ.get("HUGGING_FACE_HUB_TOKEN") and not os.environ.get("HF_TOKEN"): # HF_TOKEN is an alias
    logger.warning(
        "HUGGING_FACE_HUB_TOKEN (or HF_TOKEN) not found in environment. "
        "Accessing private/gated GGUF repos on the Hugging Face Hub might fail."
    )

def convert_messages_to_gguf_format(llama_index_messages):
    """Converts LlamaIndex ChatMessage list to llama-cpp-python expected format."""
    gguf_messages = []
    for msg in llama_index_messages:
        role = "system" # Default
        if msg.role == MessageRole.USER:
            role = "user"
        elif msg.role == MessageRole.ASSISTANT:
            role = "assistant"
        elif msg.role == MessageRole.SYSTEM:
            role = "system"
        gguf_messages.append({"role": role, "content": str(msg.content)})
    return gguf_messages

def load_gguf_model(
    model_spec: dict,
    n_gpu_layers=DEFAULT_N_GPU_LAYERS,
    n_ctx=DEFAULT_N_CTX, # This n_ctx is for loading the model
    **kwargs
):
    """
    Loads a GGUF model using llama_cpp.Llama.
    Can load from a local path or download from Hugging Face Hub.
    model_spec: dict like {'path': 'local/path.gguf'} (for local)
                       OR {'repo_id': 'TheBloke/..', 'filename': 'model.gguf'} (for Hub)
    kwargs: Additional arguments for Llama constructor or Llama.from_pretrained.
    """
    local_model_path = model_spec.get('path')
    hub_repo_id = model_spec.get('repo_id')
    hub_filename = model_spec.get('filename')

    # Filter out keys already handled by main params or not relevant to Llama constructor
    # to avoid passing them twice or causing errors.
    # verbose is a common Llama constructor arg.
    llama_constructor_kwargs = {
        k: v for k, v in kwargs.items() if k in ['verbose', 'seed', 'logits_all', 'embedding'] # Add other valid Llama args
    }


    try:
        if hub_repo_id and hub_filename:
            logger.info(f"Loading GGUF model from Hugging Face Hub: repo_id='{hub_repo_id}', filename='{hub_filename}' "
                        f"with n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}")
            llm = Llama.from_pretrained(
                repo_id=hub_repo_id,
                filename=hub_filename,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=kwargs.get('verbose', False), # Pass verbose if provided
                **llama_constructor_kwargs # Pass other filtered kwargs
            )
            logger.info(f"Successfully loaded GGUF model from Hub: {hub_repo_id}/{hub_filename}")
        elif local_model_path:
            if not os.path.exists(local_model_path):
                logger.error(f"Local GGUF model path does not exist: {local_model_path}")
                raise FileNotFoundError(f"Local GGUF model not found at {local_model_path}")
            logger.info(f"Loading GGUF model from local path: {local_model_path} "
                        f"with n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}")
            llm = Llama(
                model_path=local_model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=kwargs.get('verbose', False),
                **llama_constructor_kwargs
            )
            logger.info(f"Successfully loaded GGUF model from local path: {local_model_path}")
        else:
            raise ValueError("GGUF model_spec must contain either 'path' (for local) "
                             "or both 'repo_id' and 'filename' (for Hub).")
        return llm
    except Exception as e:
        model_identifier = f"{hub_repo_id}/{hub_filename}" if hub_repo_id and hub_filename else local_model_path
        logger.error(f"Error loading GGUF model {model_identifier}: {e}", exc_info=True)
        if "Repository Not Found" in str(e) or "401" in str(e) or "EntryNotFoundError" in str(e) or "Request header field Authorization is not allowed by Access-Control-Allow-Headers" in str(e):
             logger.error(
                 f"This might be an invalid Hugging Face repo_id/filename for GGUF, "
                 f"or a private/gated GGUF repository. Ensure HUGGING_FACE_HUB_TOKEN is set in your .env file "
                 f"and you have access to: {model_identifier}"
            )
        raise

def generate_gguf_chat_stream(
    llm_instance: Llama,
    messages: list, # Expects LlamaIndex ChatMessage objects
    temperature: float = DEFAULT_GGUF_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS, # This max_tokens is for generation
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    stop: list = None, # List of stop strings
    grammar_str: str = None, # GBNF grammar as a string
    **kwargs # For other create_chat_completion params
):
    """
    Generates a chat response stream from a loaded GGUF model.
    Yields delta content chunks.
    """
    converted_messages = convert_messages_to_gguf_format(messages)
    logger.info(f"Generating GGUF stream. Msgs: {len(converted_messages)}, Temp: {temperature}, MaxTokens: {max_tokens}")

    grammar = None
    if grammar_str:
        try:
            grammar = LlamaGrammar.from_string(grammar_str)
            logger.info("Applied GBNF grammar.")
        except Exception as e:
            logger.warning(f"Failed to parse GBNF grammar: {e}. Proceeding without grammar.")

    completion_kwargs = {
        "messages": converted_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "stream": True,
    }
    if stop:
        completion_kwargs["stop"] = stop
    if grammar:
        completion_kwargs["grammar"] = grammar

    # Add any other specific kwargs passed from the payload's 'model_specific_params'
    # These are generation parameters.
    for k, v in kwargs.items():
        if k not in completion_kwargs and v is not None:
            completion_kwargs[k] = v

    try:
        for chunk in llm_instance.create_chat_completion(**completion_kwargs):
            delta = chunk['choices'][0]['delta']
            content_chunk = delta.get('content', '')
            if content_chunk:
                yield content_chunk
            if chunk['choices'][0].get('finish_reason') is not None:
                # This indicates the end of the stream for this choice
                break
    except Exception as e:
        logger.error(f"Error during GGUF stream generation: {e}", exc_info=True)
        yield f"[LLM GGUF Error: {str(e)}]"