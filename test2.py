import json
import re
from datetime import datetime, timezone
from typing import Dict, Any
from uuid import uuid4

import pandas as pd
import streamlit as st
from PIL import Image

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


MODEL_NAME = "gemini-2.0-flash"

st.set_page_config(
    page_title="Lost & Found Intake",
    page_icon="🧳",
    layout="wide",
)

# -----------------------
# SECRETS
# -----------------------

@st.cache_resource
def get_secrets():
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


# -----------------------
# GEMINI CLIENT
# -----------------------

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


# -----------------------
# CHROMA VECTOR STORE
# -----------------------

@st.cache_resource
def get_vector_store():
    """Local Chroma DB for found items."""
    if not secrets["openai_key"]:
        st.warning("OPENAI_API_KEY is not set; semantic matching will be disabled.")
        return None
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=secrets["openai_key"])
        vs = Chroma(
            collection_name="lost_and_found",
            embedding_function=embeddings,
            persist_directory="chroma_db",  # folder in your repo
        )
        return vs
    except Exception as e:
        st.warning(f"Error creating vector store: {e}")
        return None

vector_store = get_vector_store()


# -----------------------
# SAFE GEMINI HELPERS
# -----------------------

def safe_send(chat, message_content: str, context: str = ""):
    try:
        return chat.send_message(message_content)
    except genai_errors.ClientError as e:
        label = context or "chat"
        st.error(f"Gemini ClientError during {label}: {e}")
        if getattr(e, "response_json", None):
            st.json(e.response_json)
        st.stop()


def safe_generate(full_prompt: str, context: str = ""):
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
# TAGS + SMALL HELPERS
# -----------------------

@st.cache_data
def load_tag_data():
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

tag_data = load_tag_data()
if not tag_data or gemini_client is None:
    st.stop()


def extract_field(text_block: str, field: str) -> str:
    match = re.search(rf"{field}:\s*(.*)", text_block)
    return match.group(1).strip() if match else "null"


def is_structured_record(message: str) -> bool:
    return message.strip().startswith("Subway Location:")


def standardize_description(text_block: str, tags: Dict) -> Dict:
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

        if "time" not in data or not data["time"]:
            data["time"] = datetime.now(timezone.utc).isoformat()

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


def validate_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"\d{10}", phone))


def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


# ------------- VECTOR STORE OPERATIONS ------------- #

def save_found_to_vectorstore(final_json: Dict, contact: str, image_path: str = ""):
    if vector_store is None:
        return
    text_value = final_json.get("description", "")
    if not text_value:
        return

    doc_id = f"found-{uuid4()}"
    metadata = {
        "id": doc_id,
        "subway_location": final_json.get("subway_location", []),
        "color": final_json.get("color", []),
        "item_category": final_json.get("item_category", ""),
        "item_type": final_json.get("item_type", []),
        "contact": contact,
        "image_path": image_path,
        "json_data": final_json,
    }

    vector_store.add_texts(
        texts=[text_value],
        metadatas=[metadata],
        ids=[doc_id],
    )
    vector_store.persist()


def find_matches_in_vectorstore(final_json: Dict, top_k: int = 3):
    if vector_store is None:
        return []

    query_text = final_json.get("description", "")
    if not query_text:
        return []

    # Simple metadata filter: category only (Chroma filter syntax is strict equality)
    _filter = None
    if final_json.get("item_category") and final_json["item_category"] != "null":
        _filter = {"item_category": final_json["item_category"]}

    try:
        docs_scores = vector_store.similarity_search_with_score(
            query_text,
            k=top_k,
            filter=_filter,
        )
        return docs_scores
    except Exception as e:
        st.error(f"Error during vector search: {e}")
        return []


# -----------------------
# APP NAV
# -----------------------

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

    for msg in st.session_state.operator_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
            if st.button("Save Found Item to VectorDB"):
                save_found_to_vectorstore(final_json, contact, image_path="")
                st.success("Found item saved to Chroma vector DB.")


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

    for msg in st.session_state.user_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
                    # we’re not storing lost items anywhere persistent in this version,
                    # just using the description as a query against vector DB
                    st.success("Lost item report submitted (not stored, only used for matching).")

                    with st.spinner("Searching for similar found items in VectorDB..."):
                        matches = find_matches_in_vectorstore(final_json, top_k=3)

                    if matches:
                        st.subheader("Top candidate matches")
                        for doc, score in matches:
                            meta = doc.metadata or {}
                            st.markdown(f"**Score:** `{score:.4f}`")
                            st.write(meta.get("json_data", {}).get("description", doc.page_content))
                            st.write(f"Location: {meta.get('subway_location')}")
                            st.write(f"Color: {meta.get('color')}")
                            st.write(f"Category: {meta.get('item_category')}")
                            st.write(f"Type: {meta.get('item_type')}")
                            if meta.get("image_path"):
                                st.image(meta["image_path"], width=200)
                            st.markdown("---")
                    else:
                        st.info("No close matches found in the vector DB yet.")
