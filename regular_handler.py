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
MIN_TEMP_FOR_SAMPLING = 1e-4 # Smallest strictly positive temperature

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
    model_identifier: str, 
    device_map="auto",
    torch_dtype_str="auto",
    use_bnb_4bit=False,
    trust_remote_code=True, 
    **kwargs 
):
    """Loads a regular Hugging Face model using transformers.pipeline."""
    logger.info(f"Loading Regular HF model: {model_identifier} with device_map={device_map}, "
                f"dtype={torch_dtype_str}, 4bit={use_bnb_4bit}, trust_remote_code={trust_remote_code}")

    actual_torch_dtype = get_torch_dtype(torch_dtype_str)
    quantization_config_obj = None # Renamed from quantization_config to avoid confusion
    if use_bnb_4bit:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and actual_torch_dtype == torch.bfloat16 else torch.float16
        quantization_config_obj = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        logger.info(f"Using BitsAndBytes 4-bit quantization with compute_dtype: {compute_dtype}.")

    try:
        tokenizer_load_kwargs = {"trust_remote_code": trust_remote_code}
        if HF_TOKEN:
            tokenizer_load_kwargs["token"] = HF_TOKEN
        
        tokenizer = AutoTokenizer.from_pretrained(model_identifier, **tokenizer_load_kwargs)

        pipeline_load_args = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code, 
            "torch_dtype": actual_torch_dtype,
            **kwargs # Pass any other kwargs from the function signature
        }
        
        # Only add quantization_config to pipeline_load_args if it's actually configured.
        # This prevents `quantization_config=None` from being passed and potentially stored
        # in a way that interferes with generation parameters.
        if quantization_config_obj:
            pipeline_load_args["quantization_config"] = quantization_config_obj
        
        if HF_TOKEN:
            pipeline_load_args["token"] = HF_TOKEN


        pipe = pipeline(
            "text-generation",
            model=model_identifier,
            tokenizer=tokenizer, 
            **pipeline_load_args
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
    pipe_instance, 
    messages: list, 
    temperature: float = DEFAULT_HF_TEMPERATURE,
    max_new_tokens: int = DEFAULT_HF_MAX_NEW_TOKENS,
    top_k: int = DEFAULT_HF_TOP_K,
    top_p: float = DEFAULT_HF_TOP_P,
    do_sample: bool = DEFAULT_HF_DO_SAMPLE,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    **kwargs 
):
    """
    Generates a chat response stream from a loaded Hugging Face model pipeline.
    Yields delta content chunks.
    """
    from transformers import TextIteratorStreamer 

    converted_messages = convert_messages_to_hf_format(messages)
    
    # Handle temperature and do_sample logic
    current_do_sample = do_sample
    current_temperature = temperature
    current_top_k = top_k
    current_top_p = top_p

    if current_temperature < MIN_TEMP_FOR_SAMPLING:
        if current_do_sample:
            logger.warning(
                f"Temperature is {current_temperature} (not strictly positive). "
                f"Forcing do_sample=False for greedy decoding, as requested temperature is too low for sampling."
            )
        current_do_sample = False

    if not current_do_sample: 
        current_temperature = None 
        current_top_k = None
        current_top_p = None
    
    logger.info(
        f"Generating Regular HF stream. Msgs: {len(converted_messages)}, "
        f"Temp: {current_temperature if current_do_sample else 'N/A (Greedy)'}, "
        f"DoSample: {current_do_sample}, MaxNewTokens: {max_new_tokens}"
    )

    if pipe_instance.tokenizer.chat_template is None:
        logger.warning(f"Tokenizer for {pipe_instance.model.name_or_path} does not have a chat_template. "
                       "Falling back to basic concatenation, which might be suboptimal.")
        prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in converted_messages])
        input_for_pipeline = prompt_text
    else:
        input_for_pipeline = converted_messages 

    streamer = TextIteratorStreamer(
        pipe_instance.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True 
    )

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": current_temperature,
        "top_k": current_top_k,
        "top_p": current_top_p,
        "do_sample": current_do_sample,
        "repetition_penalty": repetition_penalty,
        "streamer": streamer,
        "return_full_text": False,
        "quantization_config": None, # <--- ADDED THIS LINE
    }

    for k, v in kwargs.items():
        if k not in generation_kwargs and v is not None: 
            generation_kwargs[k] = v
    
    eos_token_str = pipe_instance.tokenizer.eos_token
    if not eos_token_str:
        logger.debug(f"EOS token not explicitly defined for tokenizer {pipe_instance.tokenizer.name_or_path}. Manual stripping might be less effective.")

    thread = Thread(target=pipe_instance, args=(input_for_pipeline,), kwargs=generation_kwargs)
    thread.start()

    try:
        full_response_text = ""
        for new_text_chunk in streamer:
            if new_text_chunk:
                cleaned_chunk = new_text_chunk
                if eos_token_str:
                    if cleaned_chunk == eos_token_str:
                        cleaned_chunk = "" 
                    elif cleaned_chunk.endswith(eos_token_str):
                        cleaned_chunk = cleaned_chunk[:-len(eos_token_str)]
                
                if cleaned_chunk: 
                    full_response_text += cleaned_chunk
                    yield cleaned_chunk

        thread.join(timeout=kwargs.get("thread_join_timeout", 60))
        if thread.is_alive():
            logger.warning(f"Generation thread for {pipe_instance.model.name_or_path} did not finish in time.")

    except Exception as e:
        logger.error(f"Error during Regular HF stream generation: {e}", exc_info=True)
        yield f"[LLM Regular Error: {str(e)}]"
    finally:
        if thread.is_alive():
            logger.warning(f"Generation thread for {pipe_instance.model.name_or_path} still alive post-iteration.")
