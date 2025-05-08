# regular_handler.py
import os
import logging
import torch
from threading import Thread
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
from llama_index.core.llms import MessageRole # Using for message structure consistency

# Constants for Regular HF model parameters
DEFAULT_HF_MAX_NEW_TOKENS = 512
DEFAULT_HF_TEMPERATURE = 0.7
DEFAULT_HF_TOP_K = 50
DEFAULT_HF_TOP_P = 0.95
DEFAULT_HF_DO_SAMPLE = True
DEFAULT_REPETITION_PENALTY = 1.1

logger = logging.getLogger(__name__)

# --- HUGGING FACE HUB TOKEN ---
HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
if HF_TOKEN:
    logger.info("Hugging Face Hub token found and will be used for downloading models/tokenizers.")
else:
    logger.warning("HUGGING_FACE_HUB_TOKEN not found in environment. Gated models may fail to load.")

def get_torch_dtype(dtype_str: str = "auto"):
    if dtype_str == "auto":
        return "auto"
    if dtype_str == "torch.float16":
        return torch.float16
    if dtype_str == "torch.bfloat16":
        return torch.bfloat16
    if dtype_str == "torch.float32":
        return torch.float32
    return "auto" # Default fallback

def convert_messages_to_hf_format(llama_index_messages):
    """Converts LlamaIndex ChatMessage list to Hugging Face pipeline expected format."""
    hf_messages = []
    for msg in llama_index_messages:
        role = "user"
        if msg.role == MessageRole.USER:
            role = "user"
        elif msg.role == MessageRole.ASSISTANT:
            role = "assistant"
        elif msg.role == MessageRole.SYSTEM:
            role = "system"
        hf_messages.append({"role": role, "content": str(msg.content)})
    return hf_messages

def load_regular_model(
    model_identifier: str, # This can be a local path or a Hugging Face Hub ID
    device_map="auto",
    torch_dtype_str="auto",
    use_bnb_4bit=False,
    trust_remote_code=True, # Default to True, can be overridden
    **kwargs # For other pipeline/model loading kwargs
):
    """Loads a regular Hugging Face model using transformers.pipeline."""
    logger.info(f"Loading Regular HF model: {model_identifier} with device_map={device_map}, "
                f"dtype={torch_dtype_str}, 4bit={use_bnb_4bit}, trust_remote_code={trust_remote_code}")

    actual_torch_dtype = get_torch_dtype(torch_dtype_str)
    quantization_config = None
    if use_bnb_4bit:
        # Ensure torch_dtype is compatible with bnb_4bit_compute_dtype
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and actual_torch_dtype == torch.bfloat16 else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        logger.info(f"Using BitsAndBytes 4-bit quantization with compute_dtype: {compute_dtype}.")
        # If quantizing, torch_dtype for the main model load should be None or auto,
        # as BNB handles the dtype of quantized layers.
        # However, `pipeline` might still want a general dtype. Setting to None if quantizing.
        # actual_torch_dtype = None # Let BNB handle it.

    try:
        tokenizer_load_kwargs = {"trust_remote_code": trust_remote_code}
        pipeline_load_kwargs = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            # Pass torch_dtype unless bnb is used, then it might be better to let bnb control.
            # Or, pass compute_dtype if that makes more sense for pipeline's expectation.
            # For simplicity, pass what was derived. BNB will override for its layers.
            "torch_dtype": actual_torch_dtype,
            "quantization_config": quantization_config,
        }
        
        # Add token for both tokenizer and model loading if available
        if HF_TOKEN:
            tokenizer_load_kwargs["token"] = HF_TOKEN
            # Pass token to model_kwargs for pipeline, so it gets to model.from_pretrained
            pipeline_load_kwargs["model_kwargs"] = pipeline_load_kwargs.get("model_kwargs", {})
            pipeline_load_kwargs["model_kwargs"]["token"] = HF_TOKEN


        # It's often better to load tokenizer first to inspect it, e.g., for chat templates
        tokenizer = AutoTokenizer.from_pretrained(model_identifier, **tokenizer_load_kwargs)

        # TextIteratorStreamer needs to be initialized for streaming
        # This is not part of model loading but generation setup.
        # from transformers import TextIteratorStreamer (moved to generate_regular_chat_stream)

        pipe = pipeline(
            "text-generation",
            model=model_identifier,
            tokenizer=tokenizer, # Pass the pre-loaded tokenizer
            **pipeline_load_kwargs
        )
        logger.info(f"Successfully loaded Regular HF model: {model_identifier}")
        return pipe
    except Exception as e:
        logger.error(f"Error loading Regular HF model {model_identifier}: {e}", exc_info=True)
        if "401 Client Error" in str(e) or "Repository Not Found" in str(e) or \
           "requires you to be authenticated" in str(e) or "UserAccessDenied" in str(e):
            logger.error(
                "This might be a gated model. Ensure HUGGING_FACE_HUB_TOKEN is set correctly "
                "in your .env file and you have accepted the model's terms on Hugging Face Hub."
            )
        raise

def generate_regular_chat_stream(
    pipe_instance, # This is the loaded transformers.pipeline object
    messages: list, # Expects LlamaIndex ChatMessage objects
    temperature: float = DEFAULT_HF_TEMPERATURE,
    max_new_tokens: int = DEFAULT_HF_MAX_NEW_TOKENS,
    top_k: int = DEFAULT_HF_TOP_K,
    top_p: float = DEFAULT_HF_TOP_P,
    do_sample: bool = DEFAULT_HF_DO_SAMPLE,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    **kwargs # For other pipeline generation_kwargs
):
    """
    Generates a chat response stream from a loaded Hugging Face model pipeline.
    Yields delta content chunks.
    """
    from transformers import TextIteratorStreamer # Import here to keep load_model clean

    converted_messages = convert_messages_to_hf_format(messages)
    logger.info(f"Generating Regular HF stream. Msgs: {len(converted_messages)}, Temp: {temperature}, MaxNewTokens: {max_new_tokens}")

    if pipe_instance.tokenizer.chat_template is None:
        logger.warning(f"Tokenizer for {pipe_instance.model.name_or_path} does not have a chat_template. "
                       "Falling back to basic concatenation, which might be suboptimal.")
        # Basic fallback: join messages. For proper non-chat-templated models, format a single prompt string.
        prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in converted_messages])
        # Add an assistant turn prompt if this is how the model expects it
        # This part is model-specific if no chat template.
        # For now, we assume the model can take the raw history.
        input_for_pipeline = prompt_text
    else:
        input_for_pipeline = converted_messages # Pipeline handles chat template

    streamer = TextIteratorStreamer(
        pipe_instance.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True # Usually, we don't want EOS/BOS tokens in the stream
    )

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature if do_sample else None, # Temp only for sampling
        "top_k": top_k if do_sample else None,
        "top_p": top_p if do_sample else None,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "streamer": streamer,
        "return_full_text": False, # Crucial for chat, ensures only new text is part of stream
    }

    # Add any other specific kwargs passed for generation
    for k, v in kwargs.items():
        if k not in generation_kwargs and v is not None: # Avoid overriding main params
            generation_kwargs[k] = v

    thread = Thread(target=pipe_instance, args=(input_for_pipeline,), kwargs=generation_kwargs)
    thread.start()

    try:
        full_response_text = ""
        for new_text_chunk in streamer:
            if new_text_chunk:
                full_response_text += new_text_chunk
                yield new_text_chunk

        thread.join(timeout=kwargs.get("thread_join_timeout", 30)) # Increased timeout
        if thread.is_alive():
            logger.warning(f"Generation thread for {pipe_instance.model.name_or_path} did not finish in time.")

    except Exception as e:
        logger.error(f"Error during Regular HF stream generation: {e}", exc_info=True)
        yield f"[LLM Regular Error: {str(e)}]"
    finally:
        if thread.is_alive():
            logger.warning(f"Generation thread for {pipe_instance.model.name_or_path} still alive post-iteration.")