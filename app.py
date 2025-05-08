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
from llama_index.core.llms import ChatMessage, MessageRole # Re-using for message structure

# Load environment variables from .env file (e.g., HUGGING_FACE_HUB_TOKEN)
load_dotenv()

# Import handlers
import gguf_handler
import regular_handler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=None) # Serve index.html manually
CORS(app)

# --- Model Configuration & Management ---
GGUF_MODELS_DIR = "models/gguf"
REGULAR_MODELS_DIR = "models/regular"
ONLINE_MODELS_CONFIG_FILE = "online_models.json" # Optional config for Hub models

# In-memory cache for loaded models to avoid reloading
# Stores {'model_id': {'instance': model_instance, 'load_info': {params_used_for_load}}}
LOADED_MODELS_CACHE = {}
MODEL_INFO_CACHE = []    # Stores list of dicts: {'id': ..., 'name': ..., 'type': ..., 'path'/'repo_id': ..., 'params': {...}}
model_load_lock = threading.Lock()

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
                    "params": {
                        "default_max_new_tokens": regular_handler.DEFAULT_HF_MAX_NEW_TOKENS,
                        "default_temperature": regular_handler.DEFAULT_HF_TEMPERATURE,
                        "default_do_sample": regular_handler.DEFAULT_HF_DO_SAMPLE,
                        "trust_remote_code_default": True, # Often needed for local complex models too
                    }
                })
                processed_ids.add(model_id)
        logger.info(f"Found local Regular HF models: {[m['name'] for m in MODEL_INFO_CACHE if m.get('source_type') == 'local' and m['type'] == 'regular']}")
    
    if not MODEL_INFO_CACHE:
        logger.warning("No models found or defined. Please add models to ./models/ directories or define in online_models.json.")


def get_model_instance(model_id: str, model_load_params: dict = None):
    """
    Retrieves a loaded model instance. If not loaded, loads it.
    model_load_params are specific to the model type for loading (e.g., n_gpu_layers for GGUF)
    and are passed from the frontend with each request.
    """
    if model_load_params is None:
        model_load_params = {}

    model_meta = next((m for m in MODEL_INFO_CACHE if m['id'] == model_id), None)
    if not model_meta:
        raise ValueError(f"Model ID {model_id} not found in available models.")

    # --- Cache Check & Invalidation (Simplified) ---
    # A more robust cache would compare all relevant model_load_params against cached_load_info.
    # For now, if n_gpu_layers (GGUF) or use_bnb_4bit (HF) changes, we force a reload by clearing cache.
    # This is a basic invalidation strategy.
    if model_id in LOADED_MODELS_CACHE:
        cached_data = LOADED_MODELS_CACHE[model_id]
        cached_load_info = cached_data.get('load_info', {})
        reload_required = False

        if model_meta['type'] == 'gguf':
            requested_gpu_layers = model_load_params.get('n_gpu_layers', gguf_handler.DEFAULT_N_GPU_LAYERS)
            if cached_load_info.get('n_gpu_layers') != requested_gpu_layers:
                logger.info(f"GGUF {model_id}: n_gpu_layers changed ({cached_load_info.get('n_gpu_layers')} -> {requested_gpu_layers}). Reloading.")
                reload_required = True
        elif model_meta['type'] == 'regular':
            requested_bnb = model_load_params.get('use_bnb_4bit', False)
            if cached_load_info.get('use_bnb_4bit') != requested_bnb:
                logger.info(f"Regular HF {model_id}: use_bnb_4bit changed ({cached_load_info.get('use_bnb_4bit')} -> {requested_bnb}). Reloading.")
                reload_required = True
            # Could add checks for trust_remote_code, torch_dtype_str if they become dynamic per request
        
        if reload_required:
            with model_load_lock: # Ensure thread-safe removal
                 LOADED_MODELS_CACHE.pop(model_id, None) # Remove to force reload
                 logger.info(f"Removed {model_id} from cache due to changed load parameters.")
        elif cached_data.get('instance'):
            logger.info(f"Using cached model instance for: {model_id}")
            return cached_data['instance']


    with model_load_lock: # Ensure only one thread tries to load the same model simultaneously
        # Double check cache after acquiring lock, another thread might have loaded it.
        if model_id in LOADED_MODELS_CACHE and LOADED_MODELS_CACHE[model_id].get('instance'):
             # Basic re-check if invalidation happened above and another thread reloaded with *new* params
            logger.info(f"Model {model_id} was loaded by another thread while waiting. Using it.")
            return LOADED_MODELS_CACHE[model_id]['instance']

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

            # n_ctx for GGUF is a loading parameter. Get from model_meta.params first, then handler default.
            # UI doesn't directly send n_ctx as a model_load_param currently, but could be added.
            n_ctx_load = model_meta.get('params', {}).get('n_ctx', gguf_handler.DEFAULT_N_CTX)
            
            gpu_layers_load = model_load_params.get('n_gpu_layers', 
                                     model_meta.get('params', {}).get('n_gpu_layers_default', gguf_handler.DEFAULT_N_GPU_LAYERS))
            actual_params_used_for_load['n_gpu_layers'] = gpu_layers_load
            actual_params_used_for_load['n_ctx'] = n_ctx_load


            instance = gguf_handler.load_gguf_model(
                gguf_spec,
                n_gpu_layers=gpu_layers_load,
                n_ctx=n_ctx_load, # Pass n_ctx for loading
                verbose=model_load_params.get('verbose', False) # Example other load param
            )
        elif model_meta['type'] == 'regular':
            # model_meta['path'] is the identifier (local path OR Hub ID)
            model_identifier_for_hf = model_meta['path']
            
            # Get default trust_remote_code from model_meta if defined, else True
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
            LOADED_MODELS_CACHE[model_id] = {'instance': instance, 'load_info': actual_params_used_for_load}
            logger.info(f"Successfully loaded and cached model: {model_id}")
        else:
            logger.error(f"Failed to load model instance for {model_id}")
            # LOADED_MODELS_CACHE.pop(model_id, None) # Ensure it's not in cache if load failed
            raise RuntimeError(f"Could not load model {model_id}")
        return instance

# --- Session Management (same as your original) ---
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

    # Basic validation
    if not all([session_id, user_prompt_text, model_id_selected]):
        missing = [k for k,v in {'session_id':session_id, 'prompt':user_prompt_text, 'model_id':model_id_selected}.items() if not v]
        error_msg = f"Missing required fields: {', '.join(missing)}"
        yield f"data: {json.dumps({'error': error_msg, 'is_final': True})}\n\n"
        return

    model_meta = next((m for m in MODEL_INFO_CACHE if m['id'] == model_id_selected), None)
    if not model_meta:
        yield f"data: {json.dumps({'error': f'Model {model_id_selected} not found.', 'is_final': True})}\n\n"
        return

    try:
        llm_instance = get_model_instance(model_id_selected, model_load_params)
        
        messages_for_llm = []
        with history_lock:
            current_history_dicts = list(session_histories.get(session_id, []))

            if system_prompt_text:
                messages_for_llm.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt_text))

            for msg_dict in current_history_dicts:
                if msg_dict.get("role") == MessageRole.SYSTEM and messages_for_llm and messages_for_llm[0].role == MessageRole.SYSTEM:
                    continue
                messages_for_llm.append(ChatMessage(role=MessageRole(msg_dict["role"]), content=str(msg_dict["content"])))
        
        user_chat_message = ChatMessage(role=MessageRole.USER, content=user_prompt_text)
        messages_for_llm.append(user_chat_message)

        logger.info(f"Session {session_id} - Calling LLM '{model_id_selected}' ({model_meta['type']}) "
                    f"with {len(messages_for_llm)} messages. Gen Params: {generation_params}")

        full_response_content = ""
        stream_iterator = None
        
        # Common generation params
        common_gen_kwargs = {
            "temperature": temperature,
            # Other params will be pulled from generation_params specific to model type
        }

        if model_meta['type'] == 'gguf':
            # Merge common with GGUF specific from generation_params
            gguf_gen_kwargs = {
                **common_gen_kwargs,
                "max_tokens": int(generation_params.get('max_tokens', gguf_handler.DEFAULT_MAX_TOKENS)),
                "top_k": int(generation_params.get('top_k', gguf_handler.DEFAULT_TOP_K)),
                "top_p": float(generation_params.get('top_p', gguf_handler.DEFAULT_TOP_P)),
                "repeat_penalty": float(generation_params.get('repeat_penalty', gguf_handler.DEFAULT_REPEAT_PENALTY)),
                "stop": generation_params.get('stop', None),
                "grammar_str": generation_params.get('grammar_str', None)
            }
            stream_iterator = gguf_handler.generate_gguf_chat_stream(
                llm_instance, messages_for_llm, **gguf_gen_kwargs
            )
        elif model_meta['type'] == 'regular':
            # Merge common with Regular HF specific from generation_params
            hf_gen_kwargs = {
                **common_gen_kwargs,
                "max_new_tokens": int(generation_params.get('max_new_tokens', regular_handler.DEFAULT_HF_MAX_NEW_TOKENS)),
                "top_k": int(generation_params.get('top_k', regular_handler.DEFAULT_HF_TOP_K)),
                "top_p": float(generation_params.get('top_p', regular_handler.DEFAULT_HF_TOP_P)),
                "do_sample": bool(generation_params.get('do_sample', regular_handler.DEFAULT_HF_DO_SAMPLE)),
                "repetition_penalty": float(generation_params.get('repetition_penalty', regular_handler.DEFAULT_REPETITION_PENALTY))
            }
            stream_iterator = regular_handler.generate_regular_chat_stream(
                llm_instance, messages_for_llm, **hf_gen_kwargs
            )
        else:
            model_type = model_meta['type']
            yield f"data: {json.dumps({'error': f'Unknown model type: {model_type}', 'is_final': True})}\n\n"
            return

        for chunk_content in stream_iterator:
            if chunk_content:
                full_response_content += chunk_content
                yield f"data: {json.dumps({'text_chunk': chunk_content, 'is_final': False})}\n\n"
        
        with history_lock:
            if session_id not in session_histories:
                 initialize_session_history(session_id)

            if system_prompt_text:
                if not session_histories[session_id] or session_histories[session_id][0].get("role") != MessageRole.SYSTEM:
                    session_histories[session_id].insert(0, {"role": MessageRole.SYSTEM, "content": system_prompt_text})
                elif session_histories[session_id][0].get("content") != system_prompt_text:
                     session_histories[session_id][0]["content"] = system_prompt_text
            
            session_histories[session_id].append({"role": MessageRole.USER, "content": user_prompt_text})
            session_histories[session_id].append({"role": MessageRole.ASSISTANT, "content": full_response_content})
            # Optional: Limit history size (consider system prompt preservation)

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

    app.run(debug=True, host='0.0.0.0', port=5000)