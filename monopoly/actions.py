"""Turn raw model text into structured game actions.

Nothing in this file calls an LLM or changes the game. It only reads text.

The reply format is borrowed from last year's negotiation project:

    Thought: private reasoning (the other player never sees this)
    Talk: what you say to the other player
    Action: [TAG] ...

Deal terms are always written from the point of view of the player who wrote them:

    [SUBMIT_DEAL] give=[Reading Railroad] give_cash=50 get=[Baltic Avenue] get_cash=0
"""
import re
from dataclasses import dataclass

from tile import PropertyTile

ALL_TAGS = ("SUBMIT_DEAL", "ACCEPT_DEAL", "REJECT_DEAL", "WALK_AWAY", "BUY", "PASS")
_TAG_RE = re.compile(r"\[(" + "|".join(ALL_TAGS) + r")\]", re.IGNORECASE)

# Which tags are allowed for each kind of decision.
BUY_ACTIONS = ("BUY", "PASS")
OPEN_ACTIONS = ("SUBMIT_DEAL", "PASS")
RESPOND_ACTIONS = ("SUBMIT_DEAL", "ACCEPT_DEAL", "REJECT_DEAL", "WALK_AWAY")

DEAL_FORMAT_HINT = (
    "Use exactly: [SUBMIT_DEAL] give=[Property, Property] give_cash=N get=[Property] get_cash=M"
)


# ---------------------------------------------------------------------------
# Reading the Thought / Talk / Action sections
# ---------------------------------------------------------------------------
def strip_think_block(text):
    """Remove <think>...</think> blocks that some models print before answering."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def extract_section(text, section):
    """Return the text after 'Thought:' / 'Talk:' / 'Action:', or None if missing."""
    # [ \t]* (not \s*) after the colon: an empty "Talk:" line must not swallow the Action line.
    pattern = re.compile(
        rf"(?:^|\n)\s*{section}[ \t]*:[ \t]*(.*?)(?=\n\s*(?:Thought|Talk|Action)\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _section_pos(text, section):
    match = re.search(rf"(?:^|\n)\s*{section}\s*:", text, re.IGNORECASE)
    return match.start() if match else None


@dataclass
class ParsedReply:
    raw: str  # cleaned reply text
    thought: str | None
    talk: str | None
    action: str | None
    format_ok: bool  # True if all required sections are present and in order


def parse_reply(raw, need_talk=True):
    """Split a model reply into Thought / Talk / Action."""
    cleaned = strip_think_block(raw or "")
    # Models sometimes decorate with markdown (**Action:**, `[BUY]`). Strip it.
    cleaned = cleaned.replace("**", "").replace("`", "")

    thought = extract_section(cleaned, "Thought")
    talk = extract_section(cleaned, "Talk")
    action = extract_section(cleaned, "Action")

    format_ok = thought is not None and action is not None and (talk is not None or not need_talk)
    if format_ok:
        t, k, a = (_section_pos(cleaned, s) for s in ("Thought", "Talk", "Action"))
        format_ok = t < a and (k is None or (t < k < a))
    return ParsedReply(cleaned, thought, talk, action, format_ok)


# ---------------------------------------------------------------------------
# Deals
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Deal:
    """A trade, from the point of view of whoever is holding it.

    give_* is what that player hands over; get_* is what they receive.
    """

    give_props: tuple = ()
    give_cash: int = 0
    get_props: tuple = ()
    get_cash: int = 0

    def flipped(self):
        """The same deal, seen by the other player."""
        return Deal(self.get_props, self.get_cash, self.give_props, self.give_cash)

    def to_offer_data(self):
        """The dict format GameManager.executeTrade already uses."""
        return {
            "giveCash": self.give_cash,
            "giveProps": list(self.give_props),
            "receiveCash": self.get_cash,
            "receiveProps": list(self.get_props),
        }

    def describe(self, board):
        return (
            f"gives {describe_side(self.give_props, self.give_cash, board)} "
            f"and gets {describe_side(self.get_props, self.get_cash, board)}"
        )

    def to_tag_text(self, board):
        give = ", ".join(board.getTile(i).name for i in self.give_props)
        get = ", ".join(board.getTile(i).name for i in self.get_props)
        return f"give=[{give}] give_cash={self.give_cash} get=[{get}] get_cash={self.get_cash}"


def describe_side(props, cash, board):
    parts = [board.getTile(i).name for i in props]
    if cash:
        parts.append(f"${cash}")
    return " + ".join(parts) if parts else "nothing"


def _norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def resolve_property(name, board):
    """Match a property name to a board index. Exact match first, then a unique partial match."""
    target = _norm(name.strip("\"' "))
    if not target:
        return None
    props = [t for t in board.tiles if isinstance(t, PropertyTile)]
    for tile in props:
        if _norm(tile.name) == target:
            return tile.index
    partial = [t for t in props if target in _norm(t.name) or _norm(t.name) in target]
    return partial[0].index if len(partial) == 1 else None


_LIST_RE = {
    "give": re.compile(r"\bgive\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "get": re.compile(r"\bget\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
}
_CASH_RE = {
    "give": re.compile(r"\bgive_cash\s*[=:]\s*\$?\s*(\d+)", re.IGNORECASE),
    "get": re.compile(r"\bget_cash\s*[=:]\s*\$?\s*(\d+)", re.IGNORECASE),
}


def parse_deal(text, board):
    """Read the deal terms out of an action line. Returns (Deal, None) or (None, error)."""
    found_any = any(r.search(text) for r in _LIST_RE.values()) or any(
        r.search(text) for r in _CASH_RE.values()
    )
    if not found_any:
        return None, "Could not read the deal terms. " + DEAL_FORMAT_HINT

    props = {}
    unknown = []
    for side, regex in _LIST_RE.items():
        ids = []
        match = regex.search(text)
        if match:
            for name in (n for n in match.group(1).split(",") if n.strip()):
                prop_id = resolve_property(name, board)
                if prop_id is None:
                    unknown.append(name.strip())
                elif prop_id not in ids:
                    ids.append(prop_id)
        props[side] = tuple(ids)
    if unknown:
        valid = ", ".join(t.name for t in board.tiles if isinstance(t, PropertyTile))
        return None, f"Unknown property name(s): {', '.join(unknown)}. Valid names: {valid}."

    cash = {}
    for side, regex in _CASH_RE.items():
        match = regex.search(text)
        cash[side] = int(match.group(1)) if match else 0

    deal = Deal(props["give"], cash["give"], props["get"], cash["get"])
    if not (deal.give_props or deal.give_cash or deal.get_props or deal.get_cash):
        return None, "The deal is empty. Offer or ask for at least one property or some cash."
    return deal, None


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
@dataclass
class ParsedAction:
    kind: str | None = None  # one of ALL_TAGS, or None if the reply was unusable
    deal: Deal | None = None
    error: str | None = None  # plain-English problem, sent back to the model for a retry


def parse_action(parsed, allowed, board):
    """Decide which action a parsed reply is asking for, and check its form."""
    text = parsed.action
    if text is None:
        # Be forgiving: the model may have skipped the "Action:" label.
        if _TAG_RE.search(parsed.raw):
            text = parsed.raw
        else:
            return ParsedAction(
                error="I could not find an Action line. Reply with Thought, Talk, then "
                "'Action: [TAG]'. Allowed tags: " + ", ".join(f"[{a}]" for a in allowed) + "."
            )

    tags = {t.upper() for t in _TAG_RE.findall(text)}
    if not tags:
        return ParsedAction(
            error="No action tag found. Allowed tags right now: "
            + ", ".join(f"[{a}]" for a in allowed) + "."
        )
    if len(tags) > 1:
        return ParsedAction(
            error="You wrote more than one action tag. Choose exactly one of: "
            + ", ".join(f"[{a}]" for a in allowed) + "."
        )

    kind = tags.pop()
    if kind not in allowed:
        hint = ""
        if kind == "BUY":
            hint = (" You can only buy a property right after landing on it. To get a property the "
                    "other player owns, propose a trade with [SUBMIT_DEAL].")
        return ParsedAction(
            error=f"[{kind}] is not allowed right now.{hint} Allowed tags: "
            + ", ".join(f"[{a}]" for a in allowed) + "."
        )

    if kind == "SUBMIT_DEAL":
        deal, error = parse_deal(text, board)
        if error:
            return ParsedAction(error=error)
        return ParsedAction(kind=kind, deal=deal)
    return ParsedAction(kind=kind)
