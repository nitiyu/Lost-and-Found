# =======================
# LOST & FOUND INTAKE SYSTEM (Chroma Vector DB)
# =======================

import json
import re
from datetime import datetime, timezone
from typing import Dict, Any, List

import pandas as pd
import streamlit as st
from PIL import Image

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# -----------------------
# BASIC CONFIG
# -----------------------

MODEL_NAME = "gemini-2.0-flash"

st.set_page_config(
    page_title="Lost & Found AI",
    page_icon="🧳",
    layout="wide",
)

# Small CSS polish
st.markdown(
    """
<style>
/* overall background */
.main {
    background-color: #F3F4F6;
}

/* cards for matches / panels */
.card {
    border-radius: 16px;
    padding: 1rem 1.2rem;
    border: 1px solid #E5E7EB;
    background-color: #FFFFFF;
    margin-bottom: 0.75rem;
}

/* card title */
.card-title {
    font-weight: 600;
    font-size: 0.98rem;
    margin-bottom: 0.25rem;
}

/* subtle text */
.card-subtle {
    font-size: 0.8rem;
    color: #6B7280;
}

/* badges for tags */
.badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    background-color: #EEF2FF;
    border: 1px solid #E0E7FF;
    font-size: 0.72rem;
    color: #4F46E5;
    margin-right: 0.25rem;
    margin-top: 0.15rem;
}

/* section titles */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------
# SECRETS / CLIENTS / VECTOR STORE
# -----------------------

@st.cache_resource
def get_secrets() -> Dict[str, Any]:
    """Load API keys and config from Streamlit secrets."""
    try:
        s = st.secrets.to_dict()
    except Exception:
        s = {}
    return {
        "raw": s,
        "gemini_key": s.get("GEMINI_API_KEY"),
        "openai_key": s.get("OPENAI_API_KEY"),
    }


secrets = get_secrets()


@st.cache_resource
def get_gemini_client():
    """Initialize Gemini client."""
    if not secrets["gemini_key"]:
        st.error("GEMINI_API_KEY is not set in Streamlit secrets.")
        return None
    try:
        return genai.Client(api_key=secrets["gemini_key"])
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None


gemini_client = get_gemini_client()


@st.cache_resource
def get_vector_store():
    """
    Create or load a Chroma vector DB using OpenAI embeddings.
    Stores all *found* items with metadata.
    """
    if not secrets["openai_key"]:
        st.warning("OPENAI_API_KEY is not set; semantic matching will be disabled.")
        return None

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=secrets["openai_key"])
        vs = Chroma(
            collection_name="lost_and_found_items",
            embedding_function=embeddings,
            persist_directory="chroma_db",  # local folder for persistence
        )
        return vs
    except Exception as e:
        st.error(f"Error creating Chroma vector store: {e}")
        return None


vector_store = get_vector_store()


# -----------------------
# SAFE GEMINI HELPERS
# -----------------------

def safe_send(chat, message_content: str, context: str = ""):
    """Wrapper around chat.send_message with clear error reporting."""
    try:
        return chat.send_message(message_content)
    except genai_errors.ClientError as e:
        label = context or "chat"
        st.error(f"Gemini ClientError during {label}: {e}")
        if getattr(e, "response_json", None):
            st.json(e.response_json)
        st.stop()


def safe_generate(full_prompt: str, context: str = ""):
    """Wrapper around generate_content with clear error reporting."""
    if gemini_client is None:
        st.error("Gemini client is not available.")
        st.stop()
    try:
        return gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
        )
    except genai_errors.ClientError as e:
        label = context or "generation"
        st.error(f"Gemini ClientError during {label}: {e}")
        if getattr(e, "response_json", None):
            st.json(e.response_json)
        st.stop()


# -----------------------
# PROMPTS
# -----------------------

GENERATOR_SYSTEM_PROMPT = """
Role:
You are a Lost & Found intake operator for a public-transit system. Your job is to gather accurate factual information
about a found item, refine the description interactively with the user, and output a single final structured record.

Behavior Rules:
1. Input Handling
The user may provide either an image or a short text description.
If an image is provided, describe visible traits such as color, material, type, size, markings, and notable features.
If text is provided, restate and cleanly summarize it in factual language.
Do not wait for confirmation before giving the first description.

2. Clarification
Ask targeted concise follow up questions to collect identifying details such as brand, condition,
writing, contents, location (station), and time found.
If the user provides a station name (for example “Times Sq”, “Queensboro Plaza”), try to identify the corresponding subway line or lines.
If multiple lines serve the station, you can mention all of them. If the station name has four or more lines, record only the station name.
If the station is unclear or unknown, set Subway Location to null.
Stop asking questions once the description is clear and specific enough.
Do not include questions or notes in the final output.

3. Finalization
When you have enough detail, output only this structured record:

Subway Location: <station or null>
Color: <dominant or user provided colors or null>
Item Category: <free text category such as Bags and Accessories, Electronics, Clothing or null>
Item Type: <free text item type such as Backpack, Phone, Jacket or null>
Description: <concise free text summary combining all verified details>
"""

USER_SIDE_GENERATOR_PROMPT = """
You are a helpful assistant for riders reporting lost items on a subway system.

Input:
The user may provide an image or a short text description of the lost item.
If an image is provided, describe what you see, including color, material, size, shape, and any markings.
If text is provided, restate the description in clean factual language.

Clarification:
Then ask two to four short follow up questions to collect details such as:
color if unclear, brand or logo, contents if it is a bag, any writing, where it was lost,
and approximate time.

When you have enough information, output only this structured record:

Subway Location: <station name or null>
Color: <color or colors or null>
Item Category: <category or null>
Item Type: <type or null>
Description: <concise factual summary>

Do not include your questions or reasoning in the final structured record.
"""

STANDARDIZER_PROMPT = """
You are the Lost and Found Data Standardizer for a public transit system.
You receive structured text from another model describing an item.
Your task is to map free text fields to standardized tag values and produce a clean JSON record.

Tag Source:
All valid standardized values are in the provided Tags Excel reference summary.
Use only those lists to choose values.

Field rules:

Subway Location:
Compare only with the Subway Location tag list.
Color:
Compare only with the Color tag list.
Item Category:
Compare only with the Item Category tag list.
Item Type:
Compare only with the Item Type tag list.

Use exact or closest textual matches from the correct list only.
If no good match exists return "null" for that field.

Input format:

Subway Location: <value or null>
Color: <value or null>
Item Category: <value or null>
Item Type: <value or null>
Description: <free text description>

Output:

Return only a JSON object of this form:

{
  "subway_location": ["<line or station>", "<line or station>"],
  "color": ["<color1>", "<color2>"],
  "item_category": "<standardized category or null>",
  "item_type": ["<type1>", "<type2>"],
  "description": "<clean description>",
  "time": "<ISO 8601 UTC timestamp>"
}

If a field has a single value it is still an array where the specification says array.
If you cannot confidently match a value, use "null" or an empty array as appropriate.

Do not output any explanation. Only output the JSON object.
"""


# -----------------------
# TAG / DATA HELPERS
# -----------------------

@st.cache_data
def load_tag_data():
    """Load Tags.xlsx and prepare tag lists."""
    try:
        df = pd.read_excel("Tags.xlsx")  # requires openpyxl
        return {
            "df": df,
            "locations": sorted(set(df["Subway Location"].dropna().astype(str))),
            "colors": sorted(set(df["Color"].dropna().astype(str))),
            "categories": sorted(set(df["Item Category"].dropna().astype(str))),
            "item_types": sorted(set(df["Item Type"].dropna().astype(str))),
        }
    except Exception as e:
        st.error(f"Error loading tag data (Tags.xlsx): {e}")
        return None


def extract_field(text_block: str, field: str) -> str:
    """Extract 'Field: value' from a structured block."""
    match = re.search(rf"{field}:\s*(.*)", text_block)
    return match.group(1).strip() if match else "null"


def is_structured_record(message: str) -> bool:
    """Detect if a message is the final structured record."""
    return message.strip().startswith("Subway Location:")


def standardize_description(text_block: str, tags: Dict) -> Dict:
    """Send structured text + tag summary to Gemini and parse JSON."""
    tags_summary = (
        "\n--- TAGS REFERENCE ---\n"
        f"Subway Location tags: {', '.join(tags['locations'][:50])}\n"
        f"Color tags: {', '.join(tags['colors'][:50])}\n"
        f"Item Category tags: {', '.join(tags['categories'][:50])}\n"
        f"Item Type tags: {', '.join(tags['item_types'][:50])}\n"
    )

    full_prompt = f"{STANDARDIZER_PROMPT}\n\nHere is the structured input to standardize:\n{text_block}\n{tags_summary}"

    response = safe_generate(full_prompt, context="standardize_description")

    try:
        cleaned = response.text.strip()
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}") + 1
        json_text = cleaned[json_start:json_end]
        data = json.loads(json_text)

        # Fill missing time
        if "time" not in data or not data["time"]:
            data["time"] = datetime.now(timezone.utc).isoformat()

        # Normalize lists
        for key in ["subway_location", "color", "item_type"]:
            if key in data and isinstance(data[key], str):
                data[key] = [data[key]]
            elif key not in data:
                data[key] = []

        if "item_category" not in data:
            data["item_category"] = "null"

        if "description" not in data:
            data["description"] = extract_field(text_block, "Description")

        return data

    except Exception:
        st.error("Model output could not be parsed as JSON. Raw output below:")
        st.text(response.text)
        return {}


# -----------------------
# SIMPLE VALIDATION HELPERS
# -----------------------

def validate_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"\d{10}", phone))


def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


# -----------------------
# VECTOR STORE HELPERS
# -----------------------

def get_next_found_id() -> int:
    """Generate a simple incremental ID for found items (for display/admin)."""
    if "next_found_id" not in st.session_state:
        st.session_state.next_found_id = 1
    nid = st.session_state.next_found_id
    st.session_state.next_found_id += 1
    return nid


def save_found_item_to_vectorstore(json_data: Dict, contact: str) -> int:
    """
    Add a found item into Chroma vector DB with metadata.
    Returns a local numeric ID.
    """
    if vector_store is None:
        st.error("Vector store is not available; cannot save found item.")
        return -1

    description = json_data.get("description", "")
    if not description:
        st.error("Found item description is empty; cannot embed.")
        return -1

    found_id = get_next_found_id()

    metadata = {
        "record_type": "found",
        "found_id": found_id,
        "subway_location": json_data.get("subway_location", []),
        "color": json_data.get("color", []),
        "item_category": json_data.get("item_category", ""),
        "item_type": json_data.get("item_type", []),
        "description": description,
        "contact": contact,
        "time": json_data.get("time"),
    }

    try:
        vector_store.add_texts(
            texts=[description],
            metadatas=[metadata],
            ids=[str(found_id)],
        )
        vector_store.persist()
        return found_id
    except Exception as e:
        st.error(f"Error saving found item to vector store: {e}")
        return -1


def search_matches_for_lost_item(final_json: Dict, top_k: int):
    """
    Use Chroma to search for similar found items.
    Returns raw (doc, score) tuples. Score is a distance (lower = better).
    """
    if vector_store is None:
        return []

    query_text = final_json.get("description", "")
    if not query_text:
        return []

    # Optional filter by category
    filter_dict: Dict[str, Any] = {"record_type": "found"}
    if final_json.get("item_category") and final_json["item_category"] != "null":
        filter_dict["item_category"] = final_json["item_category"]

    try:
        docs_scores = vector_store.similarity_search_with_score(
            query_text,
            k=top_k,
            filter=filter_dict,
        )
        return docs_scores
    except Exception as e:
        st.error(f"Error during vector search: {e}")
        return []


def get_all_found_items_as_df() -> pd.DataFrame:
    """Pull all 'found' items from Chroma for admin view."""
    if vector_store is None:
        return pd.DataFrame()

    try:
        coll = vector_store._collection  # underlying chroma collection
        data = coll.get()  # ids, documents, metadatas
    except Exception as e:
        st.error(f"Error reading from vector store: {e}")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    for id_, doc, meta in zip(ids, docs, metas):
        if not meta:
            continue
        if meta.get("record_type") != "found":
            continue
        rows.append(
            {
                "found_id": meta.get("found_id", id_),
                "description": meta.get("description", doc),
                "subway_location": ", ".join(meta.get("subway_location", [])),
                "color": ", ".join(meta.get("color", [])),
                "item_category": meta.get("item_category", ""),
                "item_type": ", ".join(meta.get("item_type", [])),
                "contact": meta.get("contact", ""),
                "time": meta.get("time", ""),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# -----------------------
# APP INIT
# -----------------------

tag_data = load_tag_data()
if not tag_data:
    st.stop()

if gemini_client is None:
    st.stop()

st.sidebar.title("Lost & Found AI")
page = st.sidebar.radio(
    "View",
    [
        "Report Lost Item (User)",
        "Upload Found Item (Staff)",
        "Admin: Inventory",
    ],
)


# ===============================================================
# STAFF INTAKE
# ===============================================================

if page == "Upload Found Item (Staff)":
    st.title("Staff Intake – Upload Found Item")

    if "operator_chat" not in st.session_state:
        st.session_state.operator_chat = gemini_client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=GENERATOR_SYSTEM_PROMPT,
            ),
        )
        st.session_state.operator_msgs = []

    # left/right layout
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="section-title">Describe the found item</div>',
                    unsafe_allow_html=True)

        for msg in st.session_state.operator_msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Initial intake
        if not st.session_state.operator_msgs:
            col1, col2 = st.columns(2)
            with col1:
                uploaded_image = st.file_uploader(
                    "Image (optional)",
                    type=["jpg", "jpeg", "png"],
                    key="operator_image",
                )
            with col2:
                initial_text = st.text_input(
                    "Short description",
                    placeholder="For example: black backpack with a NASA patch",
                    key="operator_text",
                )

            if st.button("Start Intake"):
                if not uploaded_image and not initial_text:
                    st.error("Please upload an image or enter a short description.")
                else:
                    message_content = ""
                    if uploaded_image:
                        img = Image.open(uploaded_image).convert("RGB")
                        st.image(img, width=220)
                        message_content += "I have uploaded an image of the found item. "
                    if initial_text:
                        message_content += initial_text

                    st.session_state.operator_msgs.append(
                        {"role": "user", "content": message_content}
                    )
                    with st.spinner("Analyzing item"):
                        response = safe_send(
                            st.session_state.operator_chat,
                            message_content,
                            context="operator intake",
                        )
                    st.session_state.operator_msgs.append(
                        {"role": "model", "content": response.text}
                    )
                    st.rerun()

        # Continue chat
        operator_input = st.chat_input("Add more details or say 'done' when ready")
        if operator_input:
            st.session_state.operator_msgs.append(
                {"role": "user", "content": operator_input}
            )
            with st.spinner("Processing"):
                response = safe_send(
                    st.session_state.operator_chat,
                    operator_input,
                    context="operator follow-up",
                )
            st.session_state.operator_msgs.append(
                {"role": "model", "content": response.text}
            )
            st.rerun()

    with right:
        st.markdown('<div class="section-title">Standardized record</div>',
                    unsafe_allow_html=True)

        if st.session_state.operator_msgs and is_structured_record(
            st.session_state.operator_msgs[-1]["content"]
        ):
            structured_text = st.session_state.operator_msgs[-1]["content"]
            st.code(structured_text)

            final_json = standardize_description(structured_text, tag_data)
            if final_json:
                st.json(final_json)

                contact = st.text_input("Operator contact / badge")

                if st.button("Save to inventory"):
                    found_id = save_found_item_to_vectorstore(final_json, contact)
                    if found_id > 0:
                        st.success(f"Saved as inventory ID `{found_id}`")


# ===============================================================
# USER SIDE – CHAT + MATCHING NAV
# ===============================================================

if page == "Report Lost Item (User)":
    st.title("Lost & Found AI — Demo")

    # reset button (top-right)
    reset_col, _ = st.columns([1, 4])
    with reset_col:
        if st.button("Reset session"):
            for key in [
                "user_chat",
                "user_msgs",
                "lost_final_json",
                "lost_matches",
                "match_threshold",
            ]:
                st.session_state.pop(key, None)
            st.experimental_rerun()

    # initial states
    if "match_threshold" not in st.session_state:
        st.session_state.match_threshold = 20  # percent
    if "lost_matches" not in st.session_state:
        st.session_state.lost_matches = []
    if "lost_final_json" not in st.session_state:
        st.session_state.lost_final_json = None

    left, right = st.columns([3, 2])

    # ------------- LEFT: chat panel -------------
    with left:
        st.markdown('<div class="section-title">Describe Your Lost Item</div>',
                    unsafe_allow_html=True)

        # optional quick info
        with st.expander("Quick tags (optional)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                location_choice = st.selectbox(
                    "Subway station", [""] + tag_data["locations"]
                )
            with col2:
                category_choice = st.selectbox(
                    "Item category", [""] + tag_data["categories"]
                )
            with col3:
                type_choice = st.selectbox(
                    "Item type", [""] + tag_data["item_types"]
                )

        # image + short text
        col_img, col_text = st.columns(2)
        with col_img:
            uploaded_image = st.file_uploader(
                "Photo (optional)",
                type=["jpg", "jpeg", "png"],
                key="user_image",
            )
        with col_text:
            initial_text = st.text_input(
                "Short description",
                placeholder="For example: blue iPhone with cracked screen",
                key="user_text",
            )

        # chat state
        if "user_chat" not in st.session_state:
            st.session_state.user_chat = gemini_client.chats.create(
                model=MODEL_NAME,
                config=types.GenerateContentConfig(
                    system_instruction=USER_SIDE_GENERATOR_PROMPT,
                ),
            )
            st.session_state.user_msgs = []

        # show history
        for msg in st.session_state.user_msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Start report
        if not st.session_state.user_msgs and st.button("Start"):
            if not uploaded_image and not initial_text:
                st.error("Please upload an image or enter a short description.")
            else:
                message_text = ""
                if uploaded_image:
                    image = Image.open(uploaded_image).convert("RGB")
                    st.image(image, width=230)
                    message_text += "I have uploaded an image of my lost item. "
                if initial_text:
                    message_text += initial_text

                st.session_state.user_msgs.append(
                    {"role": "user", "content": message_text}
                )
                with st.spinner("Analyzing your item..."):
                    response = safe_send(
                        st.session_state.user_chat,
                        message_text,
                        context="user initial report",
                    )
                st.session_state.user_msgs.append(
                    {"role": "model", "content": response.text}
                )
                st.rerun()

        # continue chat
        user_input = st.chat_input("Answer questions here or say 'done' when ready")
        if user_input:
            st.session_state.user_msgs.append(
                {"role": "user", "content": user_input}
            )
            with st.spinner("Thinking..."):
                response = safe_send(
                    st.session_state.user_chat,
                    user_input,
                    context="user follow-up",
                )
            st.session_state.user_msgs.append(
                {"role": "model", "content": response.text}
            )
            st.rerun()

        # build final JSON + trigger search
        if st.session_state.user_msgs and is_structured_record(
            st.session_state.user_msgs[-1]["content"]
        ):
            structured_text = st.session_state.user_msgs[-1]["content"]

            merged_text = f"""
Subway Location: {location_choice or extract_field(structured_text, 'Subway Location')}
Color: {extract_field(structured_text, 'Color')}
Item Category: {category_choice or extract_field(structured_text, 'Item Category')}
Item Type: {type_choice or extract_field(structured_text, 'Item Type')}
Description: {extract_field(structured_text, 'Description')}
            """

            st.markdown("##### Final record (before tag standardization)")
            st.code(merged_text)

            final_json = standardize_description(merged_text, tag_data)
            if final_json:
                st.session_state.lost_final_json = final_json

                with st.expander("Standardized tags", expanded=False):
                    st.json(final_json)

                st.markdown("##### Contact (for staff to reach you)")
                contact = st.text_input("Phone number, ten digits")
                email = st.text_input("Email address")

                st.markdown("##### Matching settings")
                top_k = st.slider(
                    "How many candidate items to retrieve (top-K)?",
                    min_value=1,
                    max_value=10,
                    value=5,
                )

                if st.button("Find matches"):
                    if not validate_phone(contact):
                        st.error("Please enter a ten digit phone number (no spaces).")
                    elif not validate_email(email):
                        st.error("Please enter a valid email address.")
                    else:
                        if vector_store is None:
                            st.info(
                                "Vector store is not configured, so no matches can be shown yet."
                            )
                            st.session_state.lost_matches = []
                        else:
                            with st.spinner("Searching inventory for similar items..."):
                                matches = search_matches_for_lost_item(
                                    final_json, top_k=top_k
                                )
                            st.session_state.lost_matches = matches
                            st.success("Search complete! Adjust the match slider on the right.")


    # ------------- RIGHT: match threshold + results -------------
    with right:
        st.markdown('<div class="section-title">Match threshold</div>',
                    unsafe_allow_html=True)

        st.session_state.match_threshold = st.slider(
            "",
            min_value=0,
            max_value=100,
            value=st.session_state.match_threshold,
            step=5,
            format="%d%%",
            help="Only show matches at or above this similarity.",
        )

        st.markdown('<div class="section-title" style="margin-top:0.5rem;">Top Matches</div>',
                    unsafe_allow_html=True)

        matches = st.session_state.lost_matches or []

        if not matches:
            st.write("No matches above threshold.")
        else:
            shown_any = False
            thr = st.session_state.match_threshold

            for doc, score in matches:
                meta = doc.metadata or {}
                distance = float(score)
                similarity = max(0.0, 1.0 - distance)
                similarity_pct = similarity * 100.0

                if similarity_pct < thr:
                    continue

                shown_any = True

                title = meta.get("description", doc.page_content)[:90] + "…"
                loc = ", ".join(meta.get("subway_location", []))
                color = ", ".join(meta.get("color", []))
                cat = meta.get("item_category", "")
                types = ", ".join(meta.get("item_type", []))

                st.markdown(
                    f"""
<div class="card">
  <div class="card-title">{title}</div>
  <div class="card-subtle">
    Distance: <code>{distance:.4f}</code> ·
    Approx. similarity: <b>{similarity_pct:.1f}%</b>
  </div>
  <div style="margin-top:0.4rem;">
    {"".join(
        f'<span class="badge">{b}</span>'
        for b in [
            f"location: {loc}" if loc else "",
            f"color: {color}" if color else "",
            f"category: {cat}" if cat else "",
            f"type: {types}" if types else "",
        ] if b
    )}
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

            if not shown_any:
                st.write("No matches above the current threshold — try lowering it.")


# ===============================================================
# ADMIN / INVENTORY VIEW
# ===============================================================

if page == "Admin: Inventory":
    st.title("Inventory – Stored Found Items")

    if vector_store is None:
        st.error("Vector store is not available.")
    else:
        df_found = get_all_found_items_as_df()
        if df_found.empty:
            st.info("No found items stored yet.")
        else:
            st.dataframe(df_found, use_container_width=True)
