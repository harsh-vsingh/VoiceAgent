import logging
import re
from typing import List, Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama

from src.config import settings
from src.tools.file_ops import create_file, create_directory
from src.tools.code_gen import write_code
from src.tools.summarizer import summarize_text, read_and_summarize_file

logger = logging.getLogger(__name__)

AllowedIntent = Literal[
    "create_file",
    "create_directory",
    "write_code",
    "summarize_text",
    "read_and_summarize_file",
    "general_chat",
]

class Action(BaseModel):
    intent: AllowedIntent = Field(description="Intent to execute.")
    filename: Optional[str] = None
    dirname: Optional[str] = None
    content: Optional[str] = None
    prompt: Optional[str] = None
    text: Optional[str] = None
    response: Optional[str] = None


class ActionPlan(BaseModel):
    actions: List[Action] = Field(default_factory=list)

USE_RULE_BASED_ROUTER = False

raw_llm = ChatOllama(
    model=settings.ROUTER_LLM,
    base_url=getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.0,
    num_ctx=8192,
)

structured_llm = raw_llm.with_structured_output(ActionPlan)

SYSTEM_PROMPT = """
You are an intent router for a local agent.

Return ONLY structured actions using the schema.
Allowed intents:
- create_file(filename, content)
- create_directory(dirname)
- write_code(filename, prompt)
- summarize_text(text)
- read_and_summarize_file(filename)
- general_chat(response)

Rules:
1) Prefer create_directory for "create folder/directory".
2) Prefer write_code for "write code/program/script".
3) Use read_and_summarize_file when the user asks to summarize a file path.
4) Use summarize_text only when raw text is directly provided.
5) If user says "save above summary/result", use create_file with content from recent assistant output.
6) For unclear requests, use general_chat with short clarifying response.
"""

def _last_assistant_text(history: list[BaseMessage]) -> str:
    for msg in reversed(history):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content.strip()
    return ""

def _extract_user_name(history: list[BaseMessage]) -> str:
    patterns = [
        r"\bmy name is\s+([a-zA-Z][a-zA-Z\s'-]{0,40})\b",
        r"\bi am\s+([a-zA-Z][a-zA-Z\s'-]{0,40})\b",
        r"\bi'm\s+([a-zA-Z][a-zA-Z\s'-]{0,40})\b",
    ]
    for msg in reversed(history):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            t = msg.content.strip().lower()
            for p in patterns:
                m = re.search(p, t)
                if m:
                    return " ".join(w.capitalize() for w in m.group(1).strip().split())
    return ""

def _last_created_code_file(history: list[BaseMessage]) -> str:
    rx = re.compile(r"Successfully created\s+([a-zA-Z0-9_\-./]+\.py)\b", re.IGNORECASE)
    for msg in reversed(history):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str):
            m = rx.search(msg.content)
            if m:
                return m.group(1)
    return ""

def _rule_based_parse(text: str, history: list[BaseMessage]) -> list[dict] | None:
    t = text.strip()
    low = t.lower()

    # conversational memory: name recall
    if re.search(r"\bwhat('?s| is)\s+my\s+name\b", low):
        name = _extract_user_name(history)
        if name:
            return [{"intent": "general_chat", "response": f"Your name is {name}."}]
        return [{"intent": "general_chat", "response": "I don't have your name yet."}]

    # conversational memory: last code file recall
    if re.search(r"\bwhat\s+code\s+did\s+you\s+write\s+before\b", low) or re.search(r"\bwhich\s+code\s+file\b", low):
        last_file = _last_created_code_file(history)
        if last_file:
            return [{"intent": "general_chat", "response": f"I previously created `{last_file}`."}]
        return [{"intent": "general_chat", "response": "I have not created any code file in this thread yet."}]

    # greeting with name
    m = re.search(r"\b(my name is|i am|i'm)\s+([a-zA-Z][a-zA-Z\s'-]{0,40})\b", low)
    if m:
        name = " ".join(w.capitalize() for w in m.group(2).strip().split())
        return [{"intent": "general_chat", "response": f"Nice to meet you, {name}!"}]

    # create directory
    m = re.search(r"(create|make)\s+(a\s+)?(directory|folder)\s+(named?\s+)?([a-zA-Z0-9_\-./]+)", low)
    if m:
        return [{"intent": "create_directory", "dirname": m.group(5)}]

    # create file
    m = re.search(r"(create|make)\s+(a\s+)?file\s+(named?|called)?\s*([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)", low)
    if m:
        return [{"intent": "create_file", "filename": m.group(4), "content": ""}]

    # write code
    m = re.search(r"(write|generate)\s+code\s+(of|for)\s+(.+)", low)
    if m:
        topic = m.group(3).strip()
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", topic).strip("_") or "generated_code"
        return [{"intent": "write_code", "filename": f"{safe}.py", "prompt": topic}]

    # summarize file path
    m = re.search(r"^(summarize|summarise)\s+([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)$", low)
    if m:
        return [{"intent": "read_and_summarize_file", "filename": m.group(2)}]

    # summarize text
    if low.startswith("summarize ") or low.startswith("summarise "):
        payload = t.split(" ", 1)[1].strip() if " " in t else ""
        return [{"intent": "summarize_text", "text": payload}]

    # save above summary/result to file
    m = re.search(r"save\s+.*\s+to\s+([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)", low)
    if m and ("above" in low or "summary" in low or "result" in low):
        content = _last_assistant_text(history)
        if content:
            return [{"intent": "create_file", "filename": m.group(1), "content": content}]

    return None

def _normalize_actions(actions: list[dict]) -> list[dict]:
    fixed: list[dict] = []
    alias_map = {
        "create_folder": "create_directory",
        "make_directory": "create_directory",
        "chat": "general_chat",
        "summarise_text": "summarize_text",
        "summarize_file": "read_and_summarize_file",
    }

    for a in actions:
        intent = str(a.get("intent", "")).strip()
        intent = alias_map.get(intent, intent)

        if intent not in {
            "create_file",
            "create_directory",
            "write_code",
            "summarize_text",
            "read_and_summarize_file",
            "general_chat",
        }:
            fixed.append({"intent": "general_chat", "response": "Please rephrase the request."})
            continue

        if intent == "create_directory":
            dirname = a.get("dirname") or a.get("filename")
            if not dirname:
                fixed.append({"intent": "general_chat", "response": "Directory name missing."})
            else:
                fixed.append({"intent": "create_directory", "dirname": dirname})

        elif intent == "create_file":
            filename = a.get("filename")
            content = a.get("content", "")
            if not filename:
                fixed.append({"intent": "general_chat", "response": "Filename missing."})
            else:
                fixed.append({"intent": "create_file", "filename": filename, "content": content})

        elif intent == "write_code":
            filename = a.get("filename")
            prompt = a.get("prompt")
            if not filename or not prompt:
                fixed.append({"intent": "general_chat", "response": "Tell me what code you want, e.g. 'write code for binary search'."})
            else:
                fixed.append({"intent": "write_code", "filename": filename, "prompt": prompt})

        elif intent == "summarize_text":
            txt = a.get("text")
            if not txt:
                fixed.append({"intent": "general_chat", "response": "Text to summarize is missing."})
            else:
                fixed.append({"intent": "summarize_text", "text": txt})

        elif intent == "read_and_summarize_file":
            filename = a.get("filename")
            if not filename:
                fixed.append({"intent": "general_chat", "response": "Filename missing for file summary."})
            else:
                fixed.append({"intent": "read_and_summarize_file", "filename": filename})

        else:
            fixed.append({"intent": "general_chat", "response": a.get("response", "Okay.")})

    return fixed

def parse_user_input(text: str, history: list[BaseMessage]) -> list[dict]:
    try:
        # deterministic first (optional)
        if USE_RULE_BASED_ROUTER:
            rb = _rule_based_parse(text, history)
            if rb:
                return _normalize_actions(rb)

        # structured LLM path
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + history[-4:] + [HumanMessage(content=text)]
        plan_obj = structured_llm.invoke(messages)
        actions = [a.model_dump(exclude_none=True) for a in plan_obj.actions]
        actions = _normalize_actions(actions)

        return actions or [{"intent": "general_chat", "response": "Please rephrase your request."}]
    except Exception as e:
        logger.exception("Failed to parse intents: %s", e)
        return [{"intent": "general_chat", "response": "I'm sorry, I couldn't process that command format."}]

def execute_plan(plan: list[dict]) -> list[dict]:
    results: list[dict] = []
    last_output = ""

    for action in plan:
        intent = action.get("intent")
        try:
            if intent == "create_file":
                content = action.get("content", "")
                if content == "$LAST_OUTPUT":
                    content = last_output
                output = create_file(filename=action.get("filename"), content=content)

            elif intent == "create_directory":
                output = create_directory(dirname=action.get("dirname"))

            elif intent == "write_code":
                output = write_code(filename=action.get("filename"), prompt=action.get("prompt"))

            elif intent == "summarize_text":
                output = summarize_text(text=action.get("text"))

            elif intent == "read_and_summarize_file":
                output = read_and_summarize_file(filename=action.get("filename"))

            elif intent == "general_chat":
                output = action.get("response", "")

            else:
                output = f"Unknown intent: {intent}"

        except Exception as e:
            output = f"Error: {str(e)}"

        last_output = str(output)
        results.append(
            {
                "intent": intent,
                "action": action,
                "output": str(output),
            }
        )

    return results