import streamlit as st
import re
import time

# =====================================================================
# INITIAL STREAMLIT CONFIGURATION (MUST BE ABSOLUTE FIRST COMMAND)
# =====================================================================
st.set_page_config(page_title="Semantic Search Engine", layout="wide")

from audio_engine import cached_transcription_pipeline_wrapper, semantic_search_engine

st.title("🎙️ Multi-Podcast Semantic Search Engine")

with st.sidebar:
    st.header("💥 Emergency Hardware Wipe")
    hard_reset = st.checkbox("ACTIVATE ENGINE RESET GATES", value=False)
    
    if hard_reset:
        st.session_state.clear()
        st.warning("⚠️ Memory cache successfully purged. Uncheck this box to resume operation!")
        st.stop()

if "database_index" not in st.session_state:
    st.session_state.database_index = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "audio_bytes_cache" not in st.session_state:
    st.session_state.audio_bytes_cache = {}

with st.sidebar:
    st.divider()
    st.header("Search Settings")
    search_mode = st.checkbox("Use Advanced Cross-Encoder Reranker", value=True)
    threshold = st.slider("Minimum Similarity Threshold", 0.0, 1.0, 0.05, 0.05)
    max_results = st.slider("Max Search Results", 1, 10, 5, 1)
    
    st.divider()
    st.header("🔍 Filter Controls")
    strict_matching = st.checkbox("Enforce Strict Keyword Presence Match", value=True)
    
    filter_options = ["All Podcasts"] + list(st.session_state.processed_files)
    selected_filter = st.selectbox(
        "Search within specific file context:",
        options=filter_options,
        index=0
    )

uploaded_files = st.file_uploader(
    "Upload Podcast Audio Files", 
    type=["wav", "mp3", "m4a", "flac"], 
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.audio_bytes_cache:
            st.session_state.audio_bytes_cache[f.name] = f.read()
            
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if new_files:
        st.subheader("Ingestion Processing Pipeline")
        for uploaded_file in new_files:
            start_time = time.time()
            with st.status(f"Processing: *{uploaded_file.name}*...", expanded=False) as status:
                file_bytes = st.session_state.audio_bytes_cache[uploaded_file.name]
                chunks = cached_transcription_pipeline_wrapper(file_bytes, uploaded_file.name)
                st.session_state.database_index.extend(chunks)
                st.session_state.processed_files.add(uploaded_file.name)
                
                elapsed_time = time.time() - start_time
                status.update(
                    label=f"Finished: {uploaded_file.name} ({len(chunks)} blocks cached in {elapsed_time:.2f}s)", 
                    state="complete"
                )
        st.rerun()

if st.session_state.database_index:
    st.success(
        f"Successfully operationalized {len(st.session_state.processed_files)} audio source(s). "
        f"Local database index size: {len(st.session_state.database_index)} fragments."
    )
    
    search_tab, inspector_tab = st.tabs(["🔍 Global Search Engine Desk", "📚 Raw Transcript Inspector"])
    
    with search_tab:
        st.subheader("Global Knowledge Base Query")
        search_query = st.text_input("What are you looking for across your podcasts?", placeholder="e.g., What did they say about inflation?")
        
        if search_query:
            if selected_filter == "All Podcasts":
                active_segments = st.session_state.database_index
            else:
                active_segments = [doc for doc in st.session_state.database_index if doc["podcast_source"] == selected_filter]
                
            search_threshold = 0.05 if strict_matching else threshold
            
            results = semantic_search_engine(
                query_string=search_query,
                document_segments=active_segments, 
                min_threshold=search_threshold,
                top_k=max_results,
                conceptual_mode=not search_mode
            )
            
            if not results:
                st.warning("No context fragments matched your search constraints.")
            else:
                st.write("### Relevant Transcript Matches:")
                displayed_count = 0
                
                for rank, result in enumerate(results, 1):
                    source_filename = result.get('podcast_source', 'Unknown')
                    time_window_str = result['chunk_window']
                    base_seconds = result.get('raw_start_seconds', 0)
                    text_content = result.get("text", "").strip()
                    query_clean = search_query.strip().lower()
                    
                    match_pattern = r'\b' + re.escape(query_clean) + r'\w*\b'
                    is_match = bool(re.search(match_pattern, text_content.lower(), flags=re.IGNORECASE))
                    
                    acoustic_pattern = r'\b(aaron|iron|aron|ira|iranian|irans|ayran)\b'
                    if not is_match and query_clean == "iran":
                        is_match = bool(re.search(acoustic_pattern, text_content.lower(), flags=re.IGNORECASE))
                    
                    if strict_matching and not is_match:
                        continue
                    
                    displayed_count += 1
                    if displayed_count > max_results:
                        break
                    
                    st.markdown(f"### Match {displayed_count}")
                    st.markdown(f"**Source File:** `{source_filename}`")
                    st.markdown(f"* **Aggregated Timeline Window:** `{time_window_str}`")
                    st.caption(f"*(Match Quality Score: {result.get('similarity_score', 0.0)}*)")
                    
                    highlight_regex = r'(' + re.escape(search_query) + r'\w*|aaron|iron|aron|ira)'
                    highlighted_text = re.sub(
                        highlight_regex, 
                        r'**\1**', 
                        text_content, 
                        flags=re.IGNORECASE
                    )
                    
                    st.write(f'"{highlighted_text}"')
                    st.write("")
                    
                    if source_filename in st.session_state.audio_bytes_cache:
                        audio_placeholder = st.empty()
                        audio_placeholder.audio(
                            st.session_state.audio_bytes_cache[source_filename],
                            format="audio/mp3",
                            start_time=int(base_seconds)
                        )
                
                if displayed_count == 0:
                    st.warning("No context fragments matched your active keyword constraints.")
                    
    with inspector_tab:
        st.subheader("📚 Global Database Transcript Matrix")
        st.info("Reassembling fragments into an un-chunked continuous script view.")
        
        sorted_db = sorted(st.session_state.database_index, key=lambda x: (x["podcast_source"], x["raw_start_seconds"]))
        
        current_file = None
        full_transcript_accumulator = []
        
        for chunk in sorted_db:
            if chunk["podcast_source"] != current_file:
                if current_file is not None:
                    st.markdown("### 📄 Reconstructed Document Script")
                    st.write(" ".join(full_transcript_accumulator))
                    st.divider()
                    full_transcript_accumulator = []
                
                current_file = chunk["podcast_source"]
                st.markdown(f"## 📁 Source File: `{current_file}`")
                
                if current_file in st.session_state.audio_bytes_cache:
                    st.markdown("#### 🎧 Media Player Desk")
                    st.audio(
                        st.session_state.audio_bytes_cache[current_file],
                        format="audio/mp3"
                    )
                    st.caption("Track audio dynamically along with chronological timeline stamps embedded below:")
                st.divider()
            
            # Formats clean structural paragraph separations on each accumulation step
            time_marker = f" \n\n**[{chunk['chunk_window']}]** "
            full_transcript_accumulator.append(time_marker + chunk["text"])
            
        if current_file is not None:
            st.markdown("### 📄 Reconstructed Document Script")
            st.write(" ".join(full_transcript_accumulator))
else:
    st.info("👋 Upload one or more podcast audio files above to get started!")