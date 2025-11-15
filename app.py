from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import requests
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datetime import datetime, timezone
import dateparser
from dateparser.search import search_dates
from nltk.stem import PorterStemmer


BASE_URL = "https://november7-730026606190.europe-west1.run.app/messages"

app = FastAPI(title="Member QA Service")


class AnswerResponse(BaseModel):
    answer: str


# -------------------- BASIC TEXT UTILS --------------------

QUESTION_WORDS = {
    "what", "when", "how", "who", "whom", "whose", "where", "why", "which",
    "did", "does", "do", "is", "are", "was", "were", "will", "shall", "should",
    "would", "could", "can", "may", "might", "have", "has", "had"
}

stemmer = PorterStemmer()

FAILURE_ANSWER = "Sorry, I couldn't infer that from the member's messages."
USER_NOT_FOUND_ANSWER = "Sorry, I couldn't identify which member your question is about."


def normalize(text: str) -> List[str]:
    """Lowercase, tokenize, and stem words."""
    tokens = re.findall(r"\w+", text.lower())
    return [stemmer.stem(t) for t in tokens]


def is_when_question(question: str) -> bool:
    q = question.lower()
    return "when" in q or "what day" in q or "what date" in q


def is_how_many_question(question: str) -> bool:
    return "how many" in question.lower()


def is_list_question(question: str, user_name: str) -> bool:
    """
    Detect list-style questions using patterns and plural nouns.
    """
    q_lower = question.lower()

    # Keyword signals of a list
    if "what are " in q_lower:
        return True
    if "which " in q_lower:
        return True
    if "list of" in q_lower:
        return True
    if "favorite" in q_lower or "favourite" in q_lower:
        return True
    if "favorites" in q_lower or "favourites" in q_lower:
        return True

    # Plural noun heuristic
    q_clean = re.sub(r"'s\b", "", q_lower)
    tokens = re.findall(r"\w+", q_clean)
    name_tokens = set(normalize(user_name))

    for t in tokens:
        if (
            len(t) > 3
            and t.endswith("s")
            and t not in QUESTION_WORDS
            and stemmer.stem(t) not in name_tokens
        ):
            return True

    return False


# -------------------- FETCH ALL MESSAGES (ROBUST) --------------------

def fetch_messages(page_size: int = 500, max_total: int = 5000) -> List[Dict[str, Any]]:
    """
    Fetch messages from /messages using pagination.
    Handles occasional 404 on high skip as "no more pages".
    Retries each page up to 3 times to avoid cold-start glitches.
    """
    all_items: List[Dict[str, Any]] = []
    skip = 0

    while True:
        url = f"{BASE_URL}?skip={skip}&limit={page_size}"

        attempt = 0
        while attempt < 3:
            try:
                resp = requests.get(url, timeout=20)

                # Sometimes the backend returns 404 when skip is beyond range:
                # treat as "no more pages".
                if resp.status_code == 404:
                    return all_items

                resp.raise_for_status()
                data = resp.json()
                items = data.get("items", [])
                if not isinstance(items, list):
                    raise ValueError("Bad /messages format.")

                all_items.extend(items)

                # If fewer than a full page, no more data.
                if len(items) < page_size:
                    return all_items

                skip += page_size
                break  # success, move to next page

            except Exception as e:
                attempt += 1
                if attempt >= 3:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Error fetching /messages after retries: {e}",
                    )

        if len(all_items) >= max_total:
            break

    return all_items


# -------------------- USER DETECTION --------------------

def build_user_index(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for m in messages:
        uid = m.get("user_id")
        uname = m.get("user_name")
        if uid and uname:
            index[uid] = uname
    return index


def detect_user(question: str, user_index: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify which member the question is about by overlapping normalized tokens
    with the user_name. Handles "Amira's" -> "Amira".
    """
    q_clean = re.sub(r"'s\b", "", question)
    q_tokens = set(normalize(q_clean))

    best_uid = None
    best_name = None
    best_overlap = 0

    for uid, uname in user_index.items():
        name_tokens = set(normalize(uname))
        overlap = len(q_tokens & name_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_uid = uid
            best_name = uname

    if best_overlap == 0:
        return None, None
    return best_uid, best_name


# -------------------- TEMPORAL / WHEN LOGIC --------------------

def has_temporal_words(text: str) -> bool:
    temporal_keywords = [
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "today", "tomorrow", "yesterday", "next week", "next month", "next year",
        "this week", "this weekend", "this month", "tonight",
        "starting", "beginning", "ending"
    ]
    lower = text.lower()
    return any(k in lower for k in temporal_keywords)


def parse_timestamp(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return dateparser.parse(ts)


def extract_resolved_date_from_message(text: str, timestamp: str):
    """
    Use dateparser.search to resolve phrases like "next Monday"
    relative to the message timestamp.
    """
    msg_dt = parse_timestamp(timestamp)
    if not msg_dt:
        return None

    base = msg_dt.astimezone(timezone.utc).replace(tzinfo=None)
    try:
        results = search_dates(text, settings={
            "RELATIVE_BASE": base,
            "PREFER_DATES_FROM": "future"
        })
    except Exception:
        return None

    if not results:
        return None

    phrase, resolved = results[0]
    return phrase, resolved


def select_best_temporal_message(question: str, user_messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    For a 'when' question:
    - use TF-IDF + lexical overlap + temporal bonus
    to pick the best candidate message for that user.
    """
    q_tokens = set(normalize(question)) - QUESTION_WORDS

    texts = [m.get("message", "") for m in user_messages]
    indices = [i for i, t in enumerate(texts) if t]
    texts = [texts[i] for i in indices]

    if not texts:
        return None

    corpus = texts + [question]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)

    docs_matrix = tfidf[:-1]
    q_vec = tfidf[-1]

    cos_sims = cosine_similarity(q_vec, docs_matrix).flatten()

    scores: List[float] = []
    for i, text in enumerate(texts):
        score = cos_sims[i]

        overlap = len(q_tokens & set(normalize(text)))
        score += 0.1 * overlap

        if has_temporal_words(text):
            score += 0.3

        scores.append(score)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best_idx] <= 0:
        return None

    original_idx = indices[best_idx]
    return user_messages[original_idx]


def build_temporal_answer(user_name: str, best_msg: Dict[str, Any]) -> Optional[str]:
    """
    Build a natural-language answer for a 'when' question.
    Return None if we can't resolve a concrete date.
    """
    text = best_msg.get("message", "")
    ts = best_msg.get("timestamp", "")

    temporal_info = extract_resolved_date_from_message(text, ts)
    msg_dt = parse_timestamp(ts)

    if not temporal_info or not msg_dt:
        return None

    phrase, resolved_dt = temporal_info
    weekday = resolved_dt.strftime("%A")
    resolved_str = resolved_dt.strftime("%B %d, %Y")
    sent_str = msg_dt.strftime("%B %d, %Y")

    return (
        f"{user_name}'s plans appear to be on {weekday}, {resolved_str}, "
        f"based on this message sent on {sent_str}: \"{text}\" "
        f"(containing the phrase \"{phrase}\")."
    )


# -------------------- HOW MANY / OWNERSHIP LOGIC --------------------

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "both": 2,
}

# Patterns that indicate possession / ownership in a message
POSSESSION_PATTERNS = [
    r"\bmy\b",
    r"\bour\b",
    r"\bmine\b",
    r"\bi own\b",
    r"\bi owned\b",
    r"\bwe own\b",
    r"\bi have\b",
    r"\bi've got\b",
    r"\bwe have\b",
    r"\bi bought\b",
    r"\bi purchased\b",
    r"\bi acquired\b",
]


def get_count_entity_from_question(question: str) -> Tuple[Optional[str], Optional[str]]:
    """
    From "How many cars does Vikram Desai have?"
    -> returns ("car", "cars")
    """
    tokens = re.findall(r"\w+", question.lower())
    for i in range(len(tokens) - 2):
        if tokens[i] == "how" and tokens[i + 1] == "many":
            raw_entity = tokens[i + 2]
            return stemmer.stem(raw_entity), raw_entity
    return None, None


def message_has_possession(text: str) -> bool:
    """Check if this message linguistically suggests ownership."""
    lower = text.lower()
    for pat in POSSESSION_PATTERNS:
        if re.search(pat, lower):
            return True
    return False


def collect_entity_synonyms(
    entity_stem: str,
    user_messages: List[Dict[str, Any]],
    user_name: str
) -> set:
    """
    Auto-learn synonyms for an entity stem from this user's messages.
    """
    name_tokens = set(normalize(user_name))
    cooccurrence: Dict[str, int] = defaultdict(int)

    for m in user_messages:
        text = m.get("message", "")
        tokens = normalize(text)

        if entity_stem not in tokens:
            continue

        for tok in tokens:
            if tok == entity_stem:
                continue
            if tok in QUESTION_WORDS:
                continue
            if tok in name_tokens:
                continue
            cooccurrence[tok] += 1

    synonyms = {tok for tok, cnt in cooccurrence.items() if cnt >= 1}
    return synonyms


def infer_entity_ownership_count(
    entity_stem: str,
    user_messages: List[Dict[str, Any]],
    entity_synonyms: set
) -> Tuple[Optional[Any], bool, bool]:
    """
    Infer count for an owned entity.

    Returns:
        (count_or_contradiction, has_ownership_evidence, has_entity_mentions)

        - count_or_contradiction:
            * integer if we can safely infer a number (0 or explicit).
            * dict with type="contradiction" and evidence list if conflicting counts.
            * None if ownership exists but no numeric count is stated.
        - has_ownership_evidence:
            True if any message indicates ownership of the entity.
        - has_entity_mentions:
            True if any message even mentions the entity (owned or not).
    """
    has_ownership = False
    has_entity_mentions = False
    explicit_evidence: List[Dict[str, Any]] = []

    for m in user_messages:
        text = m.get("message", "")
        tokens = normalize(text)
        token_set = set(tokens)

        # Decide if this message is even relevant to the entity
        is_relevant = False
        if entity_stem in token_set:
            is_relevant = True
        elif token_set & entity_synonyms:
            is_relevant = True

        if not is_relevant:
            continue

        has_entity_mentions = True

        if not message_has_possession(text):
            # mentions entity/synonym but no ownership words
            continue

        has_ownership = True

        # Look for numeric token right before the entity or any synonym
        for i, tok in enumerate(tokens):
            if tok == entity_stem or tok in entity_synonyms:
                count = None
                if i > 0:
                    prev = tokens[i - 1]
                    if prev.isdigit():
                        count = int(prev)
                    elif prev in NUMBER_WORDS:
                        count = NUMBER_WORDS[prev]
                if count is not None:
                    ts_raw = m.get("timestamp", "")
                    ts = parse_timestamp(ts_raw)
                    explicit_evidence.append(
                        {
                            "count": count,
                            "timestamp": ts,
                            "message": text,
                        }
                    )

    if explicit_evidence:
        unique_counts = {e["count"] for e in explicit_evidence}
        # All numeric evidence agrees
        if len(unique_counts) == 1:
            return list(unique_counts)[0], True, has_entity_mentions
        # Conflicting numeric evidence -> return structured contradiction object
        else:
            contradiction_obj = {
                "type": "contradiction",
                "evidence": explicit_evidence,
            }
            return contradiction_obj, True, has_entity_mentions

    if has_ownership:
        # We know they own at least one, but no numeric count
        return None, True, has_entity_mentions

    # No ownership evidence at all; if there were mentions,
    # caller will treat this as count=0; if no mentions, caller
    # will treat as "cannot infer".
    return 0, False, has_entity_mentions


def build_how_many_answer(
    question: str, user_name: str, user_messages: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Answer 'How many X does NAME have?'

    Rules:
    - If NO message even mentions the entity or any learned synonym -> return None (cannot infer).
    - If messages mention the entity/synonyms BUT never show ownership -> return count = 0.
    - If ownership + explicit consistent numeric count -> return that number.
    - If ownership but no numeric count -> cannot infer (return None).
    - If ownership + conflicting numeric counts -> explain contradictions and
      use the most recent message’s count.
    """
    entity_stem, entity_word = get_count_entity_from_question(question)
    if not entity_stem or not entity_word:
        return None

    # Auto-learn synonyms for this entity from the dataset
    synonyms = collect_entity_synonyms(entity_stem, user_messages, user_name)

    count, has_ownership, has_entity_mentions = infer_entity_ownership_count(
        entity_stem, user_messages, synonyms
    )

    # No message even mentions the entity or any synonym -> cannot infer
    if not has_entity_mentions:
        return None

    # Messages mention the entity/synonyms but never show ownership -> infer 0
    if has_entity_mentions and not has_ownership:
        return (
            f"Based on {user_name}'s messages, there is no indication they own any "
            f"{entity_word}. I infer they have 0 {entity_word}."
        )

    # Ownership evidence + explicit numeric count (simple case)
    if has_ownership and isinstance(count, int):
        return f"Based on {user_name}'s messages, I infer they have {count} {entity_word}."

    # Ownership but contradiction in numeric evidence
    if has_ownership and isinstance(count, dict) and count.get("type") == "contradiction":
        evidence = count.get("evidence", [])
        if not evidence:
            return None

        # Sort evidence by timestamp (oldest -> newest)
        def ts_key(item: Dict[str, Any]) -> datetime:
            ts = item.get("timestamp")
            if isinstance(ts, datetime):
                return ts
            return datetime.min

        evidence_sorted = sorted(evidence, key=ts_key)

        lines = []
        for item in evidence_sorted:
            ts = item.get("timestamp")
            msg = item.get("message", "")
            c = item.get("count")
            if isinstance(ts, datetime):
                date_str = ts.strftime("%Y-%m-%d")
                lines.append(f"- {date_str}: \"{msg}\" (implies {c} {entity_word})")
            else:
                lines.append(f"- \"{msg}\" (implies {c} {entity_word})")

        latest = max(evidence_sorted, key=ts_key)
        latest_count = latest.get("count")

        explanation = (
            f"I found contradictory statements in {user_name}'s messages:\n"
            + "\n".join(lines)
        )

        if latest_count is not None:
            explanation += (
                f"\nBased on the most recent message, I infer the current count is "
                f"{latest_count} {entity_word}."
            )

        return explanation

    # Ownership but no numeric count and no contradictions -> can't answer "how many"
    return None


# -------------------- LIST / "WHAT ARE" LOGIC --------------------

def extract_plural_entities(question: str, user_name: str) -> List[str]:
    """
    Extract plural noun stems from the question to use as list topics.
    Example: "What are Amira's favorite restaurants?" -> ["restaurant"]
    """
    q_clean = re.sub(r"'s\b", "", question.lower())
    tokens = re.findall(r"\w+", q_clean)
    name_tokens = set(normalize(user_name))

    entities = set()
    for t in tokens:
        if (
            len(t) > 3
            and t.endswith("s")
            and t not in QUESTION_WORDS
            and stemmer.stem(t) not in name_tokens
        ):
            entities.add(stemmer.stem(t))
    return list(entities)


def build_list_answer(question: str, user_name: str, user_messages: List[Dict[str, Any]]) -> Optional[str]:
    """
    For list-style questions:
    - Detect plural entities.
    - Collect all user messages that mention at least one of those entities.
    - If none -> return None (cannot infer).
    - If some exist -> return them as a list of statements.
    """
    entities = extract_plural_entities(question, user_name)
    if not entities:
        return None

    relevant_texts: List[str] = []
    seen = set()

    for m in user_messages:
        text = m.get("message", "").strip()
        if not text:
            continue

        tokens = set(normalize(text))

        if any(e in tokens for e in entities):
            if text not in seen:
                seen.add(text)
                relevant_texts.append(text)

    if not relevant_texts:
        return None

    # Build a bullet-list style answer
    bullets = "\n".join(f"- {t}" for t in relevant_texts)
    return (
        f"From {user_name}'s messages, here are some relevant statements:\n"
        f"{bullets}"
    )


# -------------------- /ask ENDPOINT --------------------

@app.get("/ask", response_model=AnswerResponse)
def ask(question: str = Query(..., description="Your natural-language question")):
    # Step 0: Fetch all messages
    try:
        messages = fetch_messages()
    except HTTPException:
        # Pass through HTTPException from fetch_messages
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error fetching /messages: {e}")

    if not messages:
        raise HTTPException(status_code=500, detail="No messages found.")

    # Step 1: detect which member
    user_index = build_user_index(messages)
    user_id, user_name = detect_user(question, user_index)

    if not user_id or not user_name:
        return {"answer": USER_NOT_FOUND_ANSWER}

    # Step 2: gather this member's messages
    user_messages = [m for m in messages if m.get("user_id") == user_id]
    if not user_messages:
        return {"answer": FAILURE_ANSWER}

    # Step 3: dispatch based on question type

    # 3a. WHEN questions
    if is_when_question(question):
        best_msg = select_best_temporal_message(question, user_messages)
        if not best_msg:
            return {"answer": FAILURE_ANSWER}

        temporal_answer = build_temporal_answer(user_name, best_msg)
        if not temporal_answer:
            return {"answer": FAILURE_ANSWER}

        return {"answer": temporal_answer}

    # 3b. HOW MANY (ownership) questions
    if is_how_many_question(question):
        how_many_answer = build_how_many_answer(question, user_name, user_messages)
        if how_many_answer is None:
            return {"answer": FAILURE_ANSWER}
        return {"answer": how_many_answer}

    # 3c. LIST questions
    if is_list_question(question, user_name):
        list_answer = build_list_answer(question, user_name, user_messages)
        if list_answer is None:
            return {"answer": FAILURE_ANSWER}
        return {"answer": list_answer}

    # 3d. Any other question types are not handled -> fail safely
    return {"answer": FAILURE_ANSWER}
