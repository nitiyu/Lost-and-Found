# =======================
# LOST & FOUND INTAKE SYSTEM (PostgreSQL + pgvector)
# =======================

import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Any

import pandas as pd
import streamlit as st
from PIL import Image

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from langchain_openai import OpenAIEmbeddings


# -----------------------
# BASIC CONFIG
# -----------------------

MODEL_NAME = "gemini-2.0-flash"

st.set_page_config(
    page_title="Lost & Found Intake",
    page_icon="🧳",
    layout="wide",
)


# -----------------------
# SECRETS / CLIENTS / ENGINES
# -----------------------

@st.cache_resource
def get_secrets() -> Dict[str, Any]:
    try:
        s = st.secrets.to_dict()
    except Exception:
        s = {}
    return {
        "raw": s,
        "gemini_key": s.get("GEMINI_API_KEY"),
        "openai_key": s.get("OPENAI_API_KEY"),
        "pg_conn_string": s.get("PG_CONNECTION_STRING"),
    }


secrets = get_secrets()


@st.cache_resource
def get_gemini_client():
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
def get_engine():
    """SQLAlchemy engine for PostgreSQL."""
    if not secrets["pg_conn_string"]:
        st.error("PG_CONNECTION_STRING is not set in Streamlit secrets.")
        return None
    try:
        engine = create_engine(
            secrets["pg_conn_string"],
            pool_pre_ping=True,   # <--- ADD THIS HERE
        )
        return engine
    except SQLAlchemyError as e:
        st.error(f"Error creating PostgreSQL engine: {e}")
        return None

engine = get_engine()


@st.cache_resource
def get_embedder():
    """OpenAI embedder used to generate vectors stored in PostgreSQL."""
    if not secrets["openai_key"]:
        st.warning("OPENAI_API_KEY is not set; semantic matching will be disabled.")
        return None
    try:
        return OpenAIEmbeddings(openai_api_key=secrets["openai_key"])
    except Exception as e:
        st.warning(f"Error creating OpenAI embedder: {e}")
        return None


embedder = get_embedder()


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
        df = pd.read_excel("Tags.xlsx")
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
# POSTGRES DB HELPERS
# -----------------------

def init_db():
    """Create extension + found_items and lost_items tables in PostgreSQL."""
    if engine is None:
        return

    create_extension = "CREATE EXTENSION IF NOT EXISTS vector;"

    # 1536 matches text-embedding-3-small / ada-002
    create_found = """
    CREATE TABLE IF NOT EXISTS found_items (
        id SERIAL PRIMARY KEY,
        subway_location TEXT,
        color TEXT,
        item_category TEXT,
        item_type TEXT,
        description TEXT,
        contact TEXT,
        image_path TEXT,
        json_data JSONB,
        embedding VECTOR(1536)
    );
    """

    create_lost = """
    CREATE TABLE IF NOT EXISTS lost_items (
        id SERIAL PRIMARY KEY,
        description TEXT,
        contact TEXT,
        email TEXT,
        json_data JSONB,
        embedding VECTOR(1536)
    );
    """

    create_idx = """
    CREATE INDEX IF NOT EXISTS idx_found_items_embedding
    ON found_items
    USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 100);
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(create_extension))
            conn.execute(text(create_found))
            conn.execute(text(create_lost))
            conn.execute(text(create_idx))
    except SQLAlchemyError as e:
        st.error(f"Error initializing PostgreSQL tables: {e}")


def add_found_item(json_data: Dict, contact: str, image_path: str = "") -> int:
    """Insert found item in PostgreSQL and return new id."""
    if engine is None:
        st.error("PostgreSQL engine is not available.")
        return -1

    sql = """
    INSERT INTO found_items (
        subway_location, color, item_category, item_type,
        description, contact, image_path, json_data
    )
    VALUES (:subway_location, :color, :item_category, :item_type,
            :description, :contact, :image_path, :json_data)
    RETURNING id;
    """

    subway_location = ", ".join(json_data.get("subway_location", []))
    color = ", ".join(json_data.get("color", []))
    item_category = json_data.get("item_category", "")
    item_type = ", ".join(json_data.get("item_type", []))
    description = json_data.get("description", "")

    try:
        with engine.begin() as conn:
            result = conn.execute(
                text(sql),
                {
                    "subway_location": subway_location,
                    "color": color,
                    "item_category": item_category,
                    "item_type": item_type,
                    "description": description,
                    "contact": contact,
                    "image_path": image_path,
                    "json_data": json.dumps(json_data),
                },
            )
            new_id = result.scalar()
            return new_id if new_id is not None else -1
    except SQLAlchemyError as e:
        st.error(f"Error inserting found item: {e}")
        return -1


def add_lost_item(description: str, contact: str, email: str, json_data_string: str) -> int:
    """Insert lost item in PostgreSQL and return new id."""
    if engine is None:
        st.error("PostgreSQL engine is not available.")
        return -1

    sql = """
    INSERT INTO lost_items (description, contact, email, json_data)
    VALUES (:description, :contact, :email, :json_data)
    RETURNING id;
    """

    try:
        with engine.begin() as conn:
            result = conn.execute(
                text(sql),
                {
                    "description": description,
                    "contact": contact,
                    "email": email,
                    "json_data": json_data_string,
                },
            )
            new_id = result.scalar()
            return new_id if new_id is not None else -1
    except SQLAlchemyError as e:
        st.error(f"Error inserting lost item: {e}")
        return -1


def validate_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"\d{10}", phone))


def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


# -----------------------
# EMBEDDING + MATCHING HELPERS (PostgreSQL + pgvector)
# -----------------------

def _vec_to_pgvector(vec: List[float]) -> str:
    """Convert Python list[float] to pgvector literal string."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def add_found_embedding(item_id: int, description: str):
    """Compute embedding and store in found_items.embedding."""
    if embedder is None or engine is None or item_id <= 0 or not description:
        return
    try:
        vec = embedder.embed_query(description)
        vec_str = _vec_to_pgvector(vec)
        sql = "UPDATE found_items SET embedding = :emb::vector WHERE id = :id"
        with engine.begin() as conn:
            conn.execute(text(sql), {"emb": vec_str, "id": item_id})
    except Exception as e:
        st.warning(f"Could not store embedding for found item {item_id}: {e}")


def add_lost_embedding(lost_id: int, description: str):
    """Compute embedding and store in lost_items.embedding."""
    if embedder is None or engine is None or lost_id <= 0 or not description:
        return
    try:
        vec = embedder.embed_query(description)
        vec_str = _vec_to_pgvector(vec)
        sql = "UPDATE lost_items SET embedding = :emb::vector WHERE id = :id"
        with engine.begin() as conn:
            conn.execute(text(sql), {"emb": vec_str, "id": lost_id})
    except Exception as e:
        st.warning(f"Could not store embedding for lost item {lost_id}: {e}")


def find_matches_for_lost_item(final_json: Dict, top_k: int = 3):
    """
    Use pgvector directly:
    - build a query embedding from the lost item's description
    - rank found_items by embedding distance
    - optionally filter by category / color
    """
    if embedder is None or engine is None:
        return []

    desc = final_json.get("description", "")
    if not desc:
        return []

    vec = embedder.embed_query(desc)
    vec_str = _vec_to_pgvector(vec)

    filters = ["embedding IS NOT NULL"]
    params: Dict[str, Any] = {"emb": vec_str, "k": top_k}

    category = final_json.get("item_category")
    if category and category != "null":
        filters.append("item_category = :category")
        params["category"] = category

    colors = final_json.get("color") or []
    if colors:
        color_clauses = []
        for i, c in enumerate(colors):
            key = f"c{i}"
            color_clauses.append(f"color ILIKE :{key}")
            params[key] = f"%{c}%"
        filters.append("(" + " OR ".join(color_clauses) + ")")

    where_sql = "WHERE " + " AND ".join(filters)

    sql = f"""
    SELECT
        id,
        description,
        image_path,
        subway_location,
        color,
        item_category,
        item_type,
        (embedding <-> :emb::vector) AS distance
    FROM found_items
    {where_sql}
    ORDER BY embedding <-> :emb::vector
    LIMIT :k;
    """

    try:
        with engine.begin() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        return rows
    except Exception as e:
        st.error(f"Error running match query: {e}")
        return []


# -----------------------
# APP INIT
# -----------------------

tag_data = load_tag_data()
if not tag_data:
    st.stop()

if gemini_client is None:
    st.stop()

init_db()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Upload Found Item (Operator)", "Report Lost Item (User)"],
)


# ===============================================================
# OPERATOR SIDE
# ===============================================================

if page == "Upload Found Item (Operator)":
    st.title("Operator View: Upload Found Item")

    if "operator_chat" not in st.session_state:
        st.session_state.operator_chat = gemini_client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=GENERATOR_SYSTEM_PROMPT,
            ),
        )
        st.session_state.operator_msgs = []

    # Show conversation history
    for msg in st.session_state.operator_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Initial intake
    if not st.session_state.operator_msgs:
        col1, col2 = st.columns(2)
        with col1:
            uploaded_image = st.file_uploader(
                "Image of the found item (optional)",
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
                    st.image(img, width=200)
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

    # When final structured record appears
    if st.session_state.operator_msgs and is_structured_record(
        st.session_state.operator_msgs[-1]["content"]
    ):
        structured_text = st.session_state.operator_msgs[-1]["content"]
        st.markdown("### Final structured description")
        st.code(structured_text)

        final_json = standardize_description(structured_text, tag_data)
        if final_json:
            st.success("Standardized JSON")
            st.json(final_json)

            contact = st.text_input("Operator contact or badge")
            # In this version, we don't persist images to disk; image_path left blank
            if st.button("Save Found Item"):
                item_id = add_found_item(final_json, contact, image_path="")
                add_found_embedding(item_id, final_json.get("description", ""))
                st.success("Found item saved (PostgreSQL + pgvector).")


# ===============================================================
# USER SIDE
# ===============================================================

if page == "Report Lost Item (User)":
    st.title("Report Lost Item (User)")

    st.markdown("You can give quick info using dropdowns, then refine with chat.")

    with st.expander("Optional quick info"):
        col1, col2, col3 = st.columns(3)
        with col1:
            location_choice = st.selectbox(
                "Subway station (optional)", [""] + tag_data["locations"]
            )
        with col2:
            category_choice = st.selectbox(
                "Item category (optional)", [""] + tag_data["categories"]
            )
        with col3:
            type_choice = st.selectbox(
                "Item type (optional)", [""] + tag_data["item_types"]
            )

    st.subheader("Describe or show your lost item")
    col_img, col_text = st.columns(2)
    with col_img:
        uploaded_image = st.file_uploader(
            "Image of lost item (optional)",
            type=["jpg", "jpeg", "png"],
            key="user_image",
        )
    with col_text:
        initial_text = st.text_input(
            "Short description",
            placeholder="For example: blue iPhone with cracked screen",
            key="user_text",
        )

    if "user_chat" not in st.session_state:
        st.session_state.user_chat = gemini_client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=USER_SIDE_GENERATOR_PROMPT,
            ),
        )
        st.session_state.user_msgs = []

    # Show chat history
    for msg in st.session_state.user_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Start report
    if not st.session_state.user_msgs and st.button("Start Report"):
        if not uploaded_image and not initial_text:
            st.error("Please upload an image or enter a short description.")
        else:
            message_text = ""
            if uploaded_image:
                image = Image.open(uploaded_image).convert("RGB")
                st.image(image, width=250)
                message_text += "I have uploaded an image of my lost item. "
            if initial_text:
                message_text += initial_text

            st.session_state.user_msgs.append(
                {"role": "user", "content": message_text}
            )
            with st.spinner("Analyzing"):
                response = safe_send(
                    st.session_state.user_chat,
                    message_text,
                    context="user initial report",
                )
            st.session_state.user_msgs.append(
                {"role": "model", "content": response.text}
            )
            st.rerun()

    # Continue chat
    user_input = st.chat_input("Add more details or say 'done' when ready")
    if user_input:
        st.session_state.user_msgs.append(
            {"role": "user", "content": user_input}
        )
        with st.spinner("Thinking"):
            response = safe_send(
                st.session_state.user_chat,
                user_input,
                context="user follow-up",
            )
        st.session_state.user_msgs.append(
            {"role": "model", "content": response.text}
        )
        st.rerun()

    # When final structured record appears
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

        st.markdown("### Final merged record before standardization")
        st.code(merged_text)

        final_json = standardize_description(merged_text, tag_data)
        if final_json:
            st.success("Standardized record")
            st.json(final_json)

            st.markdown("### Contact information")
            contact = st.text_input("Phone number, ten digits")
            email = st.text_input("Email address")

            if st.button("Submit Lost Item Report & Find Matches"):
                if not validate_phone(contact):
                    st.error("Please enter a ten digit phone number (no spaces).")
                elif not validate_email(email):
                    st.error("Please enter a valid email address.")
                else:
                    desc = final_json.get("description", "")

                    # Save lost item
                    lost_id = add_lost_item(
                        desc,
                        contact,
                        email,
                        json.dumps(final_json),
                    )
                    add_lost_embedding(lost_id, desc)
                    st.success("Lost item report submitted.")

                    # PostgreSQL matching with pgvector
                    with st.spinner("Searching for similar found items in PostgreSQL..."):
                        matches = find_matches_for_lost_item(final_json, top_k=3)

                    if matches:
                        st.subheader("Top candidate matches")
                        for row in matches:
                            (
                                found_id,
                                f_desc,
                                f_image_path,
                                f_loc,
                                f_color,
                                f_cat,
                                f_type,
                                distance,
                            ) = row

                            st.markdown(
                                f"**Found item ID:** `{found_id}`  —  distance: `{distance:.4f}`"
                            )
                            st.write(f_desc)
                            st.write(f"Location: {f_loc}")
                            st.write(f"Color: {f_color}")
                            st.write(f"Category: {f_cat}")
                            st.write(f"Type: {f_type}")
                            if f_image_path:
                                st.image(f_image_path, width=200)
                            st.markdown("---")
                    else:
                        st.info("No close matches found in PostgreSQL yet.")

