import torch
import soundfile as sf
import numpy as np
import streamlit as st
import io
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# =====================================================================
# HARDWARE ACCELERATION SETUP: HIGH-SPEED MPS / CUDA CONFIG
# =====================================================================
if torch.cuda.is_available():
    DEVICE_ID = 0
    EMBED_DEVICE = "cuda"
    TORCH_DTYPE = torch.float16
    BATCH_SIZE = 16  
elif torch.backends.mps.is_available():
    DEVICE_ID = "mps"        
    EMBED_DEVICE = "mps"       
    TORCH_DTYPE = torch.float16 
    BATCH_SIZE = 4   
else:
    DEVICE_ID = -1
    EMBED_DEVICE = "cpu"
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    BATCH_SIZE = 1   

TARGET_SAMPLING_RATE = 16000

def load_and_resample_audio(file_path_or_buffer) -> np.ndarray:
    data, samplerate = sf.read(file_path_or_buffer)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)
    
    if samplerate != TARGET_SAMPLING_RATE:
        audio_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        duration = len(data) / samplerate
        new_length = int(duration * TARGET_SAMPLING_RATE)
        audio_tensor = torch.nn.functional.interpolate(
            audio_tensor, size=new_length, mode='linear', align_corners=False
        )
        data = audio_tensor.squeeze(0).squeeze(0).numpy()
    return data

@st.cache_resource
def load_transcription_pipeline():
    pipe_device = DEVICE_ID if isinstance(DEVICE_ID, int) else torch.device(DEVICE_ID)
    return pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small", 
        device=pipe_device,
        framework="pt",
        dtype=TORCH_DTYPE
    )

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-base-en-v1.5", device=EMBED_DEVICE)

@st.cache_resource
def load_reranker_model():
    return CrossEncoder("BAAI/bge-reranker-base", device=EMBED_DEVICE)

# =====================================================================
# UPDATED: GRANULAR ROLLING TEXT WINDOW ACCUMULATOR
# =====================================================================
def transcribe_audio_with_timestamps(audio_array: np.ndarray, podcast_name: str):
    """Batches irregular Whisper segments into crisp, shorter text intervals."""
    transcriber = load_transcription_pipeline()
    
    pipeline_output = transcriber(
        audio_array,
        chunk_length_s=30,        
        stride_length_s=6,         
        batch_size=BATCH_SIZE,     
        return_timestamps=True,   
        generate_kwargs={
            "num_beams": 1,
            "language": "en",
            "task": "transcribe",
            "no_repeat_ngram_size": 4,  
            "repetition_penalty": 1.3    
        }
    )
    
    raw_chunks = pipeline_output.get("chunks", [])
    
    formatted_chunks = []
    texts_to_embed = []
    
    current_text_buffer = []
    current_start_seconds = None
    
    # MODIFIED: Reduced target size to 25 words for tighter, shorter segment breaking
    TARGET_WORD_COUNT = 25  
    MAX_TIME_GAP_SECONDS = 15.0 # Also lowered time gap bounds to keep timestamps closer together
    
    for chunk in raw_chunks:
        chunk_text = chunk.get("text", "").strip()
        timestamps = chunk.get("timestamp", (0.0, 0.0))
        
        if not chunk_text or not timestamps or timestamps[0] is None:
            continue
            
        start_seconds = float(timestamps[0])
        
        if current_start_seconds is None:
            current_start_seconds = start_seconds
            
        current_word_pool = " ".join(current_text_buffer).split()
        time_elapsed = start_seconds - current_start_seconds
        
        if len(current_word_pool) >= TARGET_WORD_COUNT or time_elapsed >= MAX_TIME_GAP_SECONDS:
            combined_paragraph = " ".join(current_text_buffer).strip()
            if combined_paragraph:
                m, s = divmod(int(current_start_seconds), 60)
                time_window = f"{m:02d}:{s:02d}"
                
                texts_to_embed.append(combined_paragraph)
                formatted_chunks.append({
                    "podcast_source": podcast_name,
                    "chunk_window": time_window,
                    "text": combined_paragraph,
                    "raw_start_seconds": current_start_seconds
                })
            
            current_text_buffer = []
            current_start_seconds = start_seconds
            
        current_text_buffer.append(chunk_text)
        
    if current_text_buffer:
        combined_paragraph = " ".join(current_text_buffer).strip()
        if combined_paragraph:
            m, s = divmod(int(current_start_seconds), 60)
            time_window = f"{m:02d}:{s:02d}"
            texts_to_embed.append(combined_paragraph)
            formatted_chunks.append({
                "podcast_source": podcast_name,
                "chunk_window": time_window,
                "text": combined_paragraph,
                "raw_start_seconds": current_start_seconds
            })

    if DEVICE_ID == "mps":
        torch.mps.empty_cache()

    if not texts_to_embed:
        return []

    embedder = load_embedding_model()
    all_embeddings = embedder.encode(
        texts_to_embed, 
        batch_size=len(texts_to_embed), 
        convert_to_tensor=True
    )
    
    normalized_embeddings = torch.nn.functional.normalize(all_embeddings.float(), p=2, dim=1)
    for idx, vector_tensor in enumerate(normalized_embeddings):
        formatted_chunks[idx]["normalized_embedding"] = vector_tensor.cpu().numpy()

    return formatted_chunks

@st.cache_data(show_spinner=False)
def cached_transcription_pipeline_wrapper(file_bytes, podcast_name: str):
    file_buffer = io.BytesIO(file_bytes)
    audio_array = load_and_resample_audio(file_buffer)
    return transcribe_audio_with_timestamps(audio_array, podcast_name)

# =====================================================================
# ISOLATED RETRIEVAL HYBRID FUSION SELECTION ENGINE
# =====================================================================
def semantic_search_engine(query_string: str, document_segments: list, min_threshold: float = 0.3, top_k: int = 3, conceptual_mode: bool = True):
    if not query_string or not document_segments:
        return []
        
    query_clean = query_string.lower().strip()
    
    source_groups = {}
    for seg in document_segments:
        src = seg["podcast_source"]
        if src not in source_groups:
            source_groups[src] = []
        source_groups[src].append(seg)
        
    global_bm25_scores = np.zeros(len(document_segments))
    
    for src, group in source_groups.items():
        tokenized_corpus = []
        for doc in group:
            raw_words = doc["text"].lower().split()
            harmonized_words = []
            for word in raw_words:
                clean_w = word.strip(".,!?\"'()…")
                if query_clean == "iran" and clean_w in ["aaron", "iron", "aron", "ira", "iranian", "irans"]:
                    harmonized_words.append("iran")
                else:
                    harmonized_words.append(clean_w)
            tokenized_corpus.append(harmonized_words)
            
        bm25 = BM25Okapi(tokenized_corpus)
        group_scores = bm25.get_scores([query_clean])
        
        max_s = np.max(group_scores) if len(group_scores) > 0 else 0
        if max_s > 0:
            group_scores = group_scores / max_s
            
        for idx, doc in enumerate(group):
            global_idx = document_segments.index(doc)
            global_bm25_scores[global_idx] = group_scores[idx]

    embedder = load_embedding_model()
    bge_formatted_query = f"Represent this sentence for searching relevant passages: {query_string}"
    query_vector = embedder.encode(bge_formatted_query, convert_to_numpy=False)
    if isinstance(query_vector, np.ndarray):
        query_vector = torch.from_numpy(query_vector)
        
    tensor_query = torch.nn.functional.normalize(query_vector.float(), p=2, dim=0).to(EMBED_DEVICE)
    
    dense_scores = []
    for segment in document_segments:
        tensor_segment = torch.from_numpy(segment["normalized_embedding"]).float().to(EMBED_DEVICE)
        similarity_score = torch.dot(tensor_query, tensor_segment).item()
        dense_scores.append(similarity_score)

    bm25_rank_indices = np.argsort(global_bm25_scores)[::-1]
    dense_rank_indices = np.argsort(dense_scores)[::-1]
    
    rrf_scores = np.zeros(len(document_segments))
    k_constant = 60
    
    for rank, idx in enumerate(bm25_rank_indices):
        rrf_scores[idx] += 1.0 / (k_constant + rank)
    for rank, idx in enumerate(dense_rank_indices):
        rrf_scores[idx] += 1.0 / (k_constant + rank)
        
    top_candidate_indices = np.argsort(rrf_scores)[::-1][:max(top_k * 4, 20)]
    candidates = [document_segments[idx] for idx in top_candidate_indices]
    
    if not candidates:
        return []
        
    reranker = load_reranker_model()
    query_pairs = [[query_string, doc["text"]] for doc in candidates]
    rerank_scores = reranker.predict(query_pairs)
    
    final_results = []
    for idx, doc in enumerate(candidates):
        score = float(rerank_scores[idx])
        
        if conceptual_mode:
            scaled_score = round(score, 4)
        else:
            scaled_score = round(1.0 / (1.0 + np.exp(-score)), 4)
            
        if scaled_score >= min_threshold:
            res_entry = doc.copy()
            res_entry["similarity_score"] = scaled_score
            final_results.append(res_entry)
            
    final_results.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    unique_results = []
    seen_content_hashes = set()
    for res in final_results:
        text_snippet = res["text"].lower()[:20]
        if text_snippet not in seen_content_hashes:
            seen_content_hashes.add(text_snippet)
            unique_results.append(res)
            
    return unique_results[:top_k]