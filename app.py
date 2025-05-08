# app.py
import os
import json
import logging
import threading
import glob
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from uuid import uuid4
from llama_index.core.llms import ChatMessage, MessageRole

# Load environment variables from .env file (e.g., HUGGING_FACE_HUB_TOKEN)
load_dotenv()

# Import handlers
import gguf_handler
import regular_handler # Assuming you have this for other model types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=None) # Serve index.html manually
CORS(app)

# --- Model Configuration & Management ---
GGUF_MODELS_DIR = "models/gguf"
REGULAR_MODELS_DIR = "models/regular"
ONLINE_MODELS_CONFIG_FILE = "online_models.json" # Optional config for Hub models

# In-memory cache for loaded models
# For GGUF, it will store:
# {'model_id': {'instance': model_instance, 'load_info': {...}, 'lock': threading.Lock()}}
# For Regular, it will store:
# {'model_id': {'instance': model_instance, 'load_info': {...}}}
LOADED_MODELS_CACHE = {}
MODEL_INFO_CACHE = []    # Stores list of dicts: {'id': ..., 'name': ..., 'type': ..., 'path'/'repo_id': ..., 'params': {...}}
model_load_lock = threading.Lock() # Global lock for the loading process itself

def get_model_filename_no_ext(path_or_filename):
    return os.path.splitext(os.path.basename(path_or_filename))[0]

def scan_available_models():
    """Scans model directories and optional JSON config, populates MODEL_INFO_CACHE."""
    global MODEL_INFO_CACHE
    MODEL_INFO_CACHE = []
    processed_ids = set() # To avoid duplicates if defined in JSON and also found locally

    # 1. Load from online_models.json if it exists
    if os.path.exists(ONLINE_MODELS_CONFIG_FILE):
        try:
            with open(ONLINE_MODELS_CONFIG_FILE, 'r') as f:
                online_models = json.load(f)
            for model_def in online_models:
                if model_def.get("id") and model_def.get("name") and model_def.get("type"):
                    # 'path' for regular HF Hub ID, or 'repo_id'/'filename' for GGUF Hub
                    if model_def['type'] == 'regular' and not model_def.get('path'):
                        logger.warning(f"Skipping online regular model '{model_def['name']}' due to missing 'path' (Hub ID).")
                        continue
                    if model_def['type'] == 'gguf' and not (model_def.get('repo_id') and model_def.get('filename')):
                        logger.warning(f"Skipping online GGUF model '{model_def['name']}' due to missing 'repo_id' or 'filename'.")
                        continue
                    
                    model_def.setdefault('source_type', 'hub') # Mark as hub-sourced
                    model_def.setdefault('params', {}) # Ensure 'params' key exists
                    MODEL_INFO_CACHE.append(model_def)
                    processed_ids.add(model_def["id"])
                    logger.info(f"Loaded model definition from '{ONLINE_MODELS_CONFIG_FILE}': {model_def['name']}")
                else:
                    logger.warning(f"Skipping invalid model definition in '{ONLINE_MODELS_CONFIG_FILE}': {model_def}")
        except Exception as e:
            logger.error(f"Error loading '{ONLINE_MODELS_CONFIG_FILE}': {e}", exc_info=True)

    # 2. Scan local GGUF models
    if os.path.exists(GGUF_MODELS_DIR):
        for gguf_file_path in glob.glob(os.path.join(GGUF_MODELS_DIR, "*.gguf")):
            model_name_part = get_model_filename_no_ext(gguf_file_path)
            model_id = f"gguf_local_{model_name_part.replace('.', '_').replace(' ', '_')}" # Sanitize
            if model_id in processed_ids:
                logger.info(f"Local GGUF model '{model_name_part}' already defined (likely in JSON), skipping local scan entry.")
                continue
            
            MODEL_INFO_CACHE.append({
                "id": model_id,
                "name": f"{model_name_part} (GGUF Local)",
                "type": "gguf",
                "source_type": "local",
                "path": gguf_file_path, # Local path
                "params": { # Default/Info params for UI
                    "n_ctx": gguf_handler.DEFAULT_N_CTX, # This is context for loading
                    "default_max_tokens": gguf_handler.DEFAULT_MAX_TOKENS, # For generation
                    "default_temperature": gguf_handler.DEFAULT_GGUF_TEMPERATURE,
                    "default_top_k": gguf_handler.DEFAULT_TOP_K,
                    "default_top_p": gguf_handler.DEFAULT_TOP_P,
                    "default_repeat_penalty": gguf_handler.DEFAULT_REPEAT_PENALTY,
                    "n_gpu_layers_default": gguf_handler.DEFAULT_N_GPU_LAYERS,
                }
            })
            processed_ids.add(model_id)
        logger.info(f"Found local GGUF models: {[m['name'] for m in MODEL_INFO_CACHE if m.get('source_type') == 'local' and m['type'] == 'gguf']}")

    # 3. Scan local Regular Hugging Face models (directories)
    if os.path.exists(REGULAR_MODELS_DIR):
        for model_dir_name in os.listdir(REGULAR_MODELS_DIR):
            model_dir_path = os.path.join(REGULAR_MODELS_DIR, model_dir_name)
            if os.path.isdir(model_dir_path) and os.path.exists(os.path.join(model_dir_path, "config.json")):
                model_id = f"regular_local_{model_dir_name.replace('/', '_').replace('.', '_')}"
                if model_id in processed_ids:
                    logger.info(f"Local Regular model '{model_dir_name}' already defined, skipping.")
                    continue

                MODEL_INFO_CACHE.append({
                    "id": model_id,
                    "name": f"{model_dir_name} (HF Local)",
                    "type": "regular",
                    "source_type": "local",
                    "path": model_dir_path, # Local path (acts as identifier)
                    "params": { # Default/Info params for UI (adjust as needed for regular_handler)
                        "default_max_new_tokens": regular_handler.DEFAULT_HF_MAX_NEW_TOKENS if 'regular_handler' in globals() else 512,
                        "default_temperature": regular_handler.DEFAULT_HF_TEMPERATURE if 'regular_handler' in globals() else 0.7,
                        "default_do_sample": regular_handler.DEFAULT_HF_DO_SAMPLE if 'regular_handler' in globals() else True,
                        "trust_remote_code_default": True, 
                    }
                })
                processed_ids.add(model_id)
        logger.info(f"Found local Regular HF models: {[m['name'] for m in MODEL_INFO_CACHE if m.get('source_type') == 'local' and m['type'] == 'regular']}")
    
    if not MODEL_INFO_CACHE:
        logger.warning("No models found or defined. Please add models to ./models/ directories or define in online_models.json.")


def get_model_data_from_cache(model_id: str):
    """Helper to retrieve full model data (instance, lock, etc.) from cache."""
    return LOADED_MODELS_CACHE.get(model_id)

def get_model_instance(model_id: str, model_load_params: dict = None):
    """
    Retrieves a loaded model instance. If not loaded, loads it.
    Returns the model instance.
    """
    if model_load_params is None:
        model_load_params = {}

    model_meta = next((m for m in MODEL_INFO_CACHE if m['id'] == model_id), None)
    if not model_meta:
        raise ValueError(f"Model ID {model_id} not found in available models.")

    cached_data = LOADED_MODELS_CACHE.get(model_id)

    # --- Cache Check & Invalidation Logic ---
    if cached_data:
        cached_load_info = cached_data.get('load_info', {})
        reload_required = False

        if model_meta['type'] == 'gguf':
            requested_gpu_layers = model_load_params.get('n_gpu_layers', gguf_handler.DEFAULT_N_GPU_LAYERS)
            if cached_load_info.get('n_gpu_layers') != requested_gpu_layers:
                logger.info(f"GGUF {model_id}: n_gpu_layers changed ({cached_load_info.get('n_gpu_layers')} -> {requested_gpu_layers}). Reloading.")
                reload_required = True
        elif model_meta['type'] == 'regular':
            requested_bnb = model_load_params.get('use_bnb_4bit', False) # Example for regular model
            if cached_load_info.get('use_bnb_4bit') != requested_bnb:
                logger.info(f"Regular HF {model_id}: use_bnb_4bit changed ({cached_load_info.get('use_bnb_4bit')} -> {requested_bnb}). Reloading.")
                reload_required = True
        
        if reload_required:
            with model_load_lock: # Ensure thread-safe removal from cache
                 LOADED_MODELS_CACHE.pop(model_id, None) # Remove to force reload
                 logger.info(f"Removed {model_id} from cache due to changed load parameters.")
                 cached_data = None # Nullify to ensure load path is taken
        elif cached_data.get('instance'):
            logger.info(f"Using cached model instance for: {model_id}")
            return cached_data['instance'] # Return only the instance

    # If not cached_data or reload was required and it was popped
    with model_load_lock: # Global lock for the loading operation itself
        # Double check cache after acquiring lock, another thread might have loaded it.
        cached_data = LOADED_MODELS_CACHE.get(model_id)
        if cached_data and cached_data.get('instance'):
            logger.info(f"Model {model_id} was loaded by another thread while waiting. Using it.")
            return cached_data['instance']

        logger.info(f"Attempting to load model: {model_id} ({model_meta['name']}) with params: {model_load_params}")
        instance = None
        actual_params_used_for_load = model_load_params.copy() # Store what was actually used

        if model_meta['type'] == 'gguf':
            gguf_spec = {}
            if model_meta.get('source_type') == 'hub' and model_meta.get('repo_id') and model_meta.get('filename'):
                gguf_spec['repo_id'] = model_meta['repo_id']
                gguf_spec['filename'] = model_meta['filename']
            elif model_meta.get('path'): # Assumed local if path is present
                gguf_spec['path'] = model_meta['path']
            else:
                raise ValueError(f"GGUF model_meta for {model_id} is malformed.")

            n_ctx_load = model_meta.get('params', {}).get('n_ctx', gguf_handler.DEFAULT_N_CTX)
            gpu_layers_load = model_load_params.get('n_gpu_layers', 
                                     model_meta.get('params', {}).get('n_gpu_layers_default', gguf_handler.DEFAULT_N_GPU_LAYERS))
            actual_params_used_for_load['n_gpu_layers'] = gpu_layers_load
            actual_params_used_for_load['n_ctx'] = n_ctx_load

            instance = gguf_handler.load_gguf_model(
                gguf_spec,
                n_gpu_layers=gpu_layers_load,
                n_ctx=n_ctx_load,
                verbose=model_load_params.get('verbose', False)
            )
            if instance:
                instance_generation_lock = threading.Lock()
                LOADED_MODELS_CACHE[model_id] = {
                    'instance': instance,
                    'load_info': actual_params_used_for_load,
                    'lock': instance_generation_lock # Store the lock for GGUF generation
                }
                logger.info(f"Successfully loaded and cached GGUF model with generation lock: {model_id}")

        elif model_meta['type'] == 'regular':
            # model_meta['path'] is the identifier (local path OR Hub ID)
            model_identifier_for_hf = model_meta['path']
            default_trust_remote = model_meta.get('params', {}).get('trust_remote_code_default', True)
            hf_load_args = {
                "device_map": model_load_params.get('device_map', "auto"),
                "torch_dtype_str": model_load_params.get('torch_dtype_str', "auto"),
                "use_bnb_4bit": model_load_params.get('use_bnb_4bit', False),
                "trust_remote_code": model_load_params.get('trust_remote_code', default_trust_remote)
            }
            actual_params_used_for_load.update(hf_load_args)
            instance = regular_handler.load_regular_model(
                model_identifier_for_hf,
                **hf_load_args
            )
            if instance:
                LOADED_MODELS_CACHE[model_id] = { # Regular models don't get the specific 'lock' here
                    'instance': instance,
                    'load_info': actual_params_used_for_load
                }
                logger.info(f"Successfully loaded and cached regular model: {model_id}")
        
        if not instance: # If instance is still None after trying to load
            logger.error(f"Failed to load model instance for {model_id}")
            # LOADED_MODELS_CACHE.pop(model_id, None) # Ensure it's not in cache if load failed (optional)
            raise RuntimeError(f"Could not load model {model_id}")
        return instance

# --- Session Management ---
session_histories = {}
history_lock = threading.Lock()

def initialize_session_history(session_id):
    with history_lock:
        if session_id not in session_histories:
            session_histories[session_id] = []
            logger.info(f"Initialized history for new session: {session_id}")
        return session_histories[session_id]

# --- Flask Routes ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/models', methods=['GET'])
def get_models_list_route():
    if not MODEL_INFO_CACHE: # If empty on first request after start, try scanning again.
        scan_available_models()
    return jsonify(MODEL_INFO_CACHE)

@app.route('/create-session', methods=['POST'])
def create_session_route():
    try:
        session_id = str(uuid4())
        initialize_session_history(session_id)
        logger.info(f"New backend session created: {session_id}")
        return jsonify({'status': 'success', 'session_id': session_id}), 200
    except Exception as e:
        logger.error(f"Error in /create-session: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/clear-backend-history', methods=['POST'])
def clear_backend_history_route():
    data = request.json
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'status': 'error', 'message': 'session_id is required'}), 400

    with history_lock:
        if session_id in session_histories:
            session_histories[session_id] = []
            logger.info(f"Backend chat history cleared for session: {session_id}")
            message = 'Backend chat history cleared for this session.'
        else:
            initialize_session_history(session_id) # Initialize if not present
            logger.warning(f"Attempted to clear history for session not actively on backend (or new): {session_id}. Initialized empty.")
            message = 'Backend session history was not found (or was new) and is now initialized empty.'
    return jsonify({'status': 'success', 'message': message})

def generate_chat_stream_response(payload):
    session_id = payload.get('session_id')
    user_prompt_text = payload.get('prompt')
    system_prompt_text = payload.get('system_prompt', "").strip()
    model_id_selected = payload.get('model_id')
    
    temperature = float(payload.get('temperature', 0.7))
    # model_specific_params are for GENERATION
    generation_params = payload.get('model_specific_params', {})
    # model_load_params are for LOADING the model (e.g., n_gpu_layers, use_bnb_4bit)
    model_load_params = payload.get('model_load_params', {})

    # --- CORRECTED Basic validation ---
    if not all([session_id, user_prompt_text is not None, model_id_selected]): # user_prompt_text can be empty but must be present
        missing_vars_check = {'session_id': session_id, 'prompt': user_prompt_text, 'model_id': model_id_selected}
        # Check specifically for None for prompt, as empty string is allowed.
        missing = [k for k, v in missing_vars_check.items() if v is None or (isinstance(v, str) and not v.strip() and k != 'prompt')]
        if user_prompt_text is None: # Explicitly add prompt if it's None
             if 'prompt' not in missing: missing.append('prompt')
        
        if missing: # Only yield error if there are genuinely missing required fields
            missing_fields_str = ", ".join(missing)
            error_detail_msg = f"Missing required fields: {missing_fields_str}"
            yield f"data: {json.dumps({'error': error_detail_msg, 'is_final': True})}\n\n"
            return
    # --- End of corrected validation ---


    model_meta = next((m for m in MODEL_INFO_CACHE if m['id'] == model_id_selected), None)
    if not model_meta:
        yield f"data: {json.dumps({'error': f'Model {model_id_selected} not found.', 'is_final': True})}\n\n"
        return

    try:
        llm_instance = get_model_instance(model_id_selected, model_load_params)
        
        gguf_instance_lock = None
        if model_meta['type'] == 'gguf':
            cached_model_data = get_model_data_from_cache(model_id_selected)
            if cached_model_data and 'lock' in cached_model_data:
                gguf_instance_lock = cached_model_data['lock']
            else:
                logger.error(f"CRITICAL: GGUF instance lock NOT FOUND for {model_id_selected}. THIS IS UNEXPECTED if model was loaded.")
                # Decide on behavior: proceed without lock (unsafe) or return error
                # For safety, returning an error might be better if lock is expected
                # yield f"data: {json.dumps({'error': f'Internal server error: GGUF lock missing for {model_id_selected}', 'is_final': True})}\n\n"
                # return


        messages_for_llm = []
        with history_lock:
            current_history_dicts = list(session_histories.get(session_id, []))

            if system_prompt_text: # Add system prompt if provided
                messages_for_llm.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt_text))

            # Add historical messages
            for msg_dict in current_history_dicts:
                # Avoid duplicate system prompts if already added and it's the first message
                if msg_dict.get("role") == MessageRole.SYSTEM and messages_for_llm and messages_for_llm[0].role == MessageRole.SYSTEM:
                    continue 
                messages_for_llm.append(ChatMessage(role=MessageRole(msg_dict["role"]), content=str(msg_dict["content"])))
        
        # Add current user prompt
        user_chat_message = ChatMessage(role=MessageRole.USER, content=user_prompt_text)
        messages_for_llm.append(user_chat_message)

        logger.info(f"Session {session_id} - Calling LLM '{model_id_selected}' ({model_meta['type']}) "
                    f"with {len(messages_for_llm)} messages. Gen Params: {generation_params}")

        full_response_content = ""
        stream_iterator = None
        common_gen_kwargs = {"temperature": temperature}

        if model_meta['type'] == 'gguf':
            gguf_gen_kwargs = {
                **common_gen_kwargs,
                "max_tokens": int(generation_params.get('max_tokens', gguf_handler.DEFAULT_MAX_TOKENS)),
                "top_k": int(generation_params.get('top_k', gguf_handler.DEFAULT_TOP_K)),
                "top_p": float(generation_params.get('top_p', gguf_handler.DEFAULT_TOP_P)),
                "repeat_penalty": float(generation_params.get('repeat_penalty', gguf_handler.DEFAULT_REPEAT_PENALTY)),
                "stop": generation_params.get('stop', None),
                "grammar_str": generation_params.get('grammar_str', None),
                "instance_lock": gguf_instance_lock # Pass the lock to the handler
            }
            stream_iterator = gguf_handler.generate_gguf_chat_stream(
                llm_instance, messages_for_llm, **gguf_gen_kwargs
            )
        elif model_meta['type'] == 'regular':
            hf_gen_kwargs = {
                **common_gen_kwargs,
                "max_new_tokens": int(generation_params.get('max_new_tokens', regular_handler.DEFAULT_HF_MAX_NEW_TOKENS if 'regular_handler' in globals() else 512)),
                "top_k": int(generation_params.get('top_k', regular_handler.DEFAULT_HF_TOP_K if 'regular_handler' in globals() else 50)),
                "top_p": float(generation_params.get('top_p', regular_handler.DEFAULT_HF_TOP_P if 'regular_handler' in globals() else 0.95)),
                "do_sample": bool(generation_params.get('do_sample', regular_handler.DEFAULT_HF_DO_SAMPLE if 'regular_handler' in globals() else True)),
                "repetition_penalty": float(generation_params.get('repetition_penalty', regular_handler.DEFAULT_REPETITION_PENALTY if 'regular_handler' in globals() else 1.1))
            }
            stream_iterator = regular_handler.generate_regular_chat_stream(
                llm_instance, messages_for_llm, **hf_gen_kwargs
            )
        else:
            model_type = model_meta['type']
            yield f"data: {json.dumps({'error': f'Unknown model type: {model_type}', 'is_final': True})}\n\n"
            return

        for chunk_content in stream_iterator:
            if chunk_content: # Ensure chunk is not None or empty before processing
                full_response_content += chunk_content
                yield f"data: {json.dumps({'text_chunk': chunk_content, 'is_final': False})}\n\n"
        
        # Update history after successful generation
        with history_lock:
            if session_id not in session_histories: # Should be initialized, but good to check
                 initialize_session_history(session_id)

            # Persist system prompt in history if it was used for this turn
            if system_prompt_text:
                # If history is empty or first message isn't this system prompt
                if not session_histories[session_id] or \
                   not (session_histories[session_id][0].get("role") == MessageRole.SYSTEM and \
                        session_histories[session_id][0].get("content") == system_prompt_text):
                    # Check if a different system prompt is already there
                    if session_histories[session_id] and session_histories[session_id][0].get("role") == MessageRole.SYSTEM:
                        session_histories[session_id][0]["content"] = system_prompt_text # Update existing
                    else:
                        session_histories[session_id].insert(0, {"role": MessageRole.SYSTEM, "content": system_prompt_text}) # Add new
            
            session_histories[session_id].append({"role": MessageRole.USER, "content": user_prompt_text})
            session_histories[session_id].append({"role": MessageRole.ASSISTANT, "content": full_response_content})
            # Optional: Implement history size limit here

        logger.info(f"Session {session_id} - LLM full response length: {len(full_response_content)}. History size: {len(session_histories.get(session_id, []))}")
        yield f"data: {json.dumps({'full_response': full_response_content, 'is_final': True})}\n\n"

    except Exception as e:
        logger.error(f"Error in chat stream for session {session_id}, model {model_id_selected}: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': f'LLM Processing Error: {str(e)}', 'is_final': True})}\n\n"


@app.route('/chat', methods=['POST'])
def chat_route_post():
    payload = request.json
    session_id = payload.get('session_id')
    
    if session_id: # Ensure session is initialized if it's passed
        initialize_session_history(session_id)
    
    return Response(generate_chat_stream_response(payload), mimetype='text/event-stream')


if __name__ == '__main__':
    os.makedirs(GGUF_MODELS_DIR, exist_ok=True)
    os.makedirs(REGULAR_MODELS_DIR, exist_ok=True)
    
    scan_available_models() # Initial scan at startup
    if not MODEL_INFO_CACHE:
        print(f"WARNING: No models found by scanning local directories ('{GGUF_MODELS_DIR}', '{REGULAR_MODELS_DIR}') "
              f"or defined in '{ONLINE_MODELS_CONFIG_FILE}'.")
        print("Please add GGUF files to ./models/gguf/ OR Hugging Face model directories to ./models/regular/ "
              "OR define models in online_models.json and restart the server.")

    # For development and testing the GGUF lock, threaded=True is useful.
    # For production, use a WSGI server like Gunicorn.
    # If Gunicorn uses threaded workers, this GGUF lock is essential.
    # If Gunicorn uses sync workers, each worker process has its own GGUF instance & lock.
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)