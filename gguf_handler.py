# gguf_handler.py
import os
import logging
import threading # <--- Added for type hinting Lock
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
    n_ctx=DEFAULT_N_CTX,
    **kwargs
):
    """
    Loads a GGUF model using llama_cpp.Llama.
    """
    local_model_path = model_spec.get('path')
    hub_repo_id = model_spec.get('repo_id')
    hub_filename = model_spec.get('filename')

    pass_through_keys = ['seed', 'logits_all', 'embedding', 'n_threads', 'n_batch']
    llama_constructor_kwargs = {
        k: v for k, v in kwargs.items() if k in pass_through_keys and v is not None
    }

    try:
        if hub_repo_id and hub_filename:
            logger.info(f"Loading GGUF model from Hugging Face Hub: repo_id='{hub_repo_id}', filename='{hub_filename}' "
                        f"with n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}, constructor_kwargs: {llama_constructor_kwargs}")
            llm = Llama.from_pretrained(
                repo_id=hub_repo_id,
                filename=hub_filename,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=kwargs.get('verbose', False),
                **llama_constructor_kwargs
            )
            logger.info(f"Successfully loaded GGUF model from Hub: {hub_repo_id}/{hub_filename}")
        elif local_model_path:
            if not os.path.exists(local_model_path):
                logger.error(f"Local GGUF model path does not exist: {local_model_path}")
                raise FileNotFoundError(f"Local GGUF model not found at {local_model_path}")
            logger.info(f"Loading GGUF model from local path: {local_model_path} "
                        f"with n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}, constructor_kwargs: {llama_constructor_kwargs}")
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
    messages: list,
    temperature: float = DEFAULT_GGUF_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    stop: list = None,
    grammar_str: str = None,
    instance_lock: threading.Lock = None,  # <--- Added instance_lock parameter
    **kwargs
):
    """
    Generates a chat response stream from a loaded GGUF model.
    Yields delta content chunks, with logic to strip leading EOS-like tokens from the first chunk.
    Uses instance_lock to serialize generation for this specific llm_instance.
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

    for k, v in kwargs.items():
        if k not in completion_kwargs and v is not None:
            completion_kwargs[k] = v

    first_chunk_processed = False
    model_specific_eos_str = ""
    try:
        eos_token_id = llm_instance.token_eos()
        if eos_token_id != -1 and hasattr(llm_instance, 'detokenize'):
            model_specific_eos_str = llm_instance.detokenize([eos_token_id]).decode('utf-8', errors='replace')
    except Exception as e:
        logger.warning(f"Could not detokenize model's specific EOS token ID ({eos_token_id if 'eos_token_id' in locals() else 'unknown'}): {e}")

    potential_leading_tokens_to_strip = []
    if model_specific_eos_str:
        potential_leading_tokens_to_strip.append(model_specific_eos_str)
    if stop:
        for s_item in stop:
            if isinstance(s_item, str) and s_item and s_item not in potential_leading_tokens_to_strip:
                potential_leading_tokens_to_strip.append(s_item)
    common_fallback_eos_tokens = ["</s>", "<|endoftext|>", "<|im_end|>", "<|END_OF_TURN_TOKEN|>", "<s>"]
    for fb_token in common_fallback_eos_tokens:
        if fb_token not in potential_leading_tokens_to_strip:
            potential_leading_tokens_to_strip.append(fb_token)
    potential_leading_tokens_to_strip = [s for s in potential_leading_tokens_to_strip if s]
    logger.debug(f"GGUF: Will check for and strip leading sequences from first chunk: {potential_leading_tokens_to_strip}")

    # --- Acquire lock for generation if provided ---
    if instance_lock:
        logger.debug(f"GGUF stream: Attempting to acquire instance lock for generation...")
        with instance_lock: # This will block if another thread holds the lock for this instance
            logger.debug(f"GGUF stream: Instance lock acquired. Starting generation.")
            try:
                for chunk in llm_instance.create_chat_completion(**completion_kwargs):
                    delta = chunk['choices'][0]['delta']
                    content_chunk = delta.get('content', '')
                    if content_chunk:
                        if not first_chunk_processed:
                            original_chunk_for_log = content_chunk
                            for token_to_strip in potential_leading_tokens_to_strip:
                                if content_chunk.startswith(token_to_strip):
                                    content_chunk = content_chunk[len(token_to_strip):]
                                    logger.info(
                                        f"GGUF: Stripped leading '{token_to_strip.encode('unicode_escape').decode('utf-8')}' from first chunk. "
                                        f"Original: '{original_chunk_for_log[:60].encode('unicode_escape').decode('utf-8')}', "
                                        f"New: '{content_chunk[:60].encode('unicode_escape').decode('utf-8')}'"
                                    )
                                    break
                            first_chunk_processed = True
                        if content_chunk:
                            yield content_chunk
                    if chunk['choices'][0].get('finish_reason') is not None:
                        break
            except Exception as e:
                logger.error(f"Error during GGUF stream generation (with lock): {e}", exc_info=True)
                yield f"[LLM GGUF Error: {str(e)}]"
            finally:
                logger.debug(f"GGUF stream: Generation finished/exited. Lock released implicitly by 'with' statement.")
    else:
        # Fallback if no lock is provided (e.g., if called from a context not managing locks)
        # This path would NOT be thread-safe for concurrent calls to the same llm_instance.
        logger.warning("GGUF stream: No instance_lock provided. Running generation without instance-specific synchronization. THIS MAY NOT BE THREAD-SAFE FOR CONCURRENT REQUESTS TO THE SAME MODEL INSTANCE.")
        try:
            for chunk in llm_instance.create_chat_completion(**completion_kwargs):
                delta = chunk['choices'][0]['delta']
                content_chunk = delta.get('content', '')
                if content_chunk:
                    if not first_chunk_processed:
                        original_chunk_for_log = content_chunk
                        for token_to_strip in potential_leading_tokens_to_strip:
                            if content_chunk.startswith(token_to_strip):
                                content_chunk = content_chunk[len(token_to_strip):]
                                logger.info(
                                    f"GGUF: Stripped leading '{token_to_strip.encode('unicode_escape').decode('utf-8')}' from first chunk. "
                                    f"Original: '{original_chunk_for_log[:60].encode('unicode_escape').decode('utf-8')}', "
                                    f"New: '{content_chunk[:60].encode('unicode_escape').decode('utf-8')}'"
                                )
                                break
                        first_chunk_processed = True
                    if content_chunk:
                        yield content_chunk
                if chunk['choices'][0].get('finish_reason') is not None:
                    break
        except Exception as e:
            logger.error(f"Error during GGUF stream generation (no lock): {e}", exc_info=True)
            yield f"[LLM GGUF Error: {str(e)}]"