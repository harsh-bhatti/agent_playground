"""The decision loop: ask an agent for an action, check it, retry once, then act.

Used for two things:
  - ask_buy_decision: "buy or pass?"
  - run_negotiation:  a short back-and-forth between two players that can end in a trade.

The key rule: only a VALIDATED action ever changes the game. What an agent says in
"Talk" is just text the other agent reads. This is what stops an agent from "trading"
a property it does not own.

The negotiation protocol is borrowed from last year's project: players alternate turns,
a deal is proposed with [SUBMIT_DEAL], [ACCEPT_DEAL] only works on a standing offer,
and the loop ends on a deal, a walk-away, a repeated offer, or a turn cap.
"""
from dataclasses import asdict, dataclass, field

import llm_client
from actions import (
    BUY_ACTIONS,
    OPEN_ACTIONS,
    RESPOND_ACTIONS,
    describe_side,
    parse_action,
    parse_reply,
)

MAX_ATTEMPTS = 2  # the first reply, plus one correction if it was invalid

# If a model still gives an invalid reply after the correction, do something harmless.
FALLBACK = {"buy": "PASS", "open": "PASS", "respond": "REJECT_DEAL"}


@dataclass
class Turn:
    """One agent decision, kept for the research log."""

    game_turn: int
    speaker: str
    decision: str  # "buy" | "open" | "respond"
    kind: str  # the action actually taken, e.g. "SUBMIT_DEAL"
    thought: str | None = None
    talk: str | None = None
    deal: object = None
    deal_text: str | None = None
    format_ok: bool = True  # did the FIRST reply follow Thought / Talk / Action?
    attempts: int = 1
    fell_back: bool = False  # True if the replies were invalid and we used the harmless default
    error: str | None = None
    tokens: int = 0
    retry_errors: list = field(default_factory=list)  # why each earlier attempt was rejected
    first_reply: str | None = None  # the model's first reply, kept only if it had to be retried

    def to_dict(self):
        return asdict(self)


@dataclass
class NegotiationResult:
    game_turn: int
    initiator: str
    responder: str
    outcome: str = "max_turns"  # deal | walk_away | declined | max_turns | reject_loop | invalid_deal
    proposer: str | None = None
    deal_text: str | None = None  # from the proposer's point of view
    trade: dict | None = None  # value changes, filled in when a deal happens
    turns: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Asking an agent for one valid action
# ---------------------------------------------------------------------------
def get_valid_action(game, agent, speaker, listener, decision, first_call, pending=None, pending_by=None):
    """Get a legal action from an agent. Retries once with the exact problem, then falls back."""
    allowed = {"buy": BUY_ACTIONS, "open": OPEN_ACTIONS, "respond": RESPOND_ACTIONS}[decision]
    need_talk = decision != "buy"
    tokens_before = llm_client.total_tokens_used

    reply = first_call()
    attempts = 0
    first_format_ok = None
    first_reply = reply
    retry_errors = []
    while True:
        attempts += 1
        parsed = parse_reply(reply, need_talk)
        act = parse_action(parsed, allowed, game.board)
        if act.error is None:
            if act.kind == "SUBMIT_DEAL":
                act.error = game.validateTrade(speaker, listener, act.deal.to_offer_data())
            elif act.kind == "ACCEPT_DEAL" and (pending is None or pending_by == speaker.playerName):
                act.error = "There is no offer from the other player to accept."
        if first_format_ok is None:
            first_format_ok = parsed.format_ok
        if act.error is None or attempts >= MAX_ATTEMPTS:
            break
        retry_errors.append(act.error)
        reply = agent.correct(act.error)

    fell_back = act.error is not None
    kind = FALLBACK[decision] if fell_back else act.kind
    deal = None if fell_back else act.deal
    return Turn(
        game_turn=game.turnCount,
        speaker=speaker.playerName,
        decision=decision,
        kind=kind,
        thought=parsed.thought,
        talk=None if fell_back else parsed.talk,
        deal=deal,
        deal_text=deal.to_tag_text(game.board) if deal else None,
        format_ok=first_format_ok,
        attempts=attempts,
        fell_back=fell_back,
        error=act.error if fell_back else None,
        tokens=llm_client.total_tokens_used - tokens_before,
        retry_errors=retry_errors,
        first_reply=(first_reply or "")[:800] if retry_errors else None,
    )


def _print_turn(game, turn):
    if game.showThoughts and turn.thought:
        print(f"     [thought] {turn.thought}")
    if turn.talk:
        print(f'     {turn.speaker}: "{turn.talk}"')
    text = f"[{turn.kind}]"
    if turn.deal:
        text += f" {turn.deal.describe(game.board)}"
    note = ""
    if turn.fell_back:
        note = f"   (!) invalid even after a retry, used the default. Problem: {turn.error}"
    elif turn.attempts > 1:
        note = f"   (retried once because: {turn.retry_errors[0]})"
    print(f"     -> {text}{note}")


# ---------------------------------------------------------------------------
# Buy decision
# ---------------------------------------------------------------------------
def ask_buy_decision(game, agent, player, tile):
    """Ask the agent whether to buy the property it landed on. Returns True to buy."""
    opponent = next(p for p in game.players if p.playerName != player.playerName)
    print(f"  -> [DECISION] {player.playerName} considers {tile.name} (${tile.price}).")
    try:
        turn = get_valid_action(
            game, agent, player, opponent, "buy", lambda: agent.decide_buy(game, player, tile)
        )
    finally:
        agent.end_conversation()
    _print_turn(game, turn)
    game.decisionLog.append(turn.to_dict())
    return turn.kind == "BUY"


# ---------------------------------------------------------------------------
# Negotiation between two players
# ---------------------------------------------------------------------------
def _offer_message(game, speaker, talk, deal_for_listener):
    """What the listener sees: the talk plus the deal, written from THEIR point of view.

    The speaker's private Thought is never forwarded.
    """
    give = describe_side(deal_for_listener.give_props, deal_for_listener.give_cash, game.board)
    get = describe_side(deal_for_listener.get_props, deal_for_listener.get_cash, game.board)
    return (
        f'{speaker.playerName} says: "{talk or ""}"\n'
        f"{speaker.playerName} proposes a deal. YOU would give: {give}. YOU would get: {get}."
    )


def run_negotiation(game, initiator, responder, max_turns=6):
    """The initiator may propose a trade; the two players then take turns until it ends.

    A "turn" is one agent reply (so max_turns=6 means at most 6 model calls, plus retries).
    Returns a NegotiationResult.
    """
    result = NegotiationResult(
        game_turn=game.turnCount, initiator=initiator.playerName, responder=responder.playerName
    )
    agents = {p.playerName: game.agents[p.playerName] for p in (initiator, responder)}
    speaker, listener = initiator, responder
    pending = None  # the standing offer, written from the proposer's point of view
    pending_by = None
    incoming_text = None
    incoming_offer = None
    submits = {initiator.playerName: [], responder.playerName: []}

    print(f"  [NEGOTIATION] {initiator.playerName} may open a trade with {responder.playerName}.")
    try:
        for index in range(max_turns):
            agent = agents[speaker.playerName]
            if index == 0:
                decision = "open"
                first_call = lambda: agent.open_negotiation(game, speaker, listener)
            else:
                decision = "respond"
                first_call = lambda: agent.respond(game, speaker, listener, incoming_text, incoming_offer)

            turn = get_valid_action(
                game, agent, speaker, listener, decision, first_call, pending, pending_by
            )
            _print_turn(game, turn)
            result.turns.append(turn.to_dict())
            kind = turn.kind

            if kind == "PASS":
                result.outcome = "declined"
                break
            if kind == "WALK_AWAY":
                result.outcome = "walk_away"
                break
            if kind == "ACCEPT_DEAL":
                proposer = game.getPlayer(pending_by)
                trade = game.executeTrade(proposer, speaker, pending.to_offer_data())
                if trade:
                    result.outcome = "deal"
                    result.proposer = pending_by
                    result.deal_text = pending.to_tag_text(game.board)
                    result.trade = trade
                else:
                    result.outcome = "invalid_deal"
                break
            if kind == "SUBMIT_DEAL":
                pending, pending_by = turn.deal, speaker.playerName
                incoming_offer = turn.deal.flipped()
                incoming_text = _offer_message(game, speaker, turn.talk, incoming_offer)
                history = submits[speaker.playerName]
                history.append(turn.deal_text)
                if len(history) >= 3 and len(set(history[-3:])) == 1:
                    result.outcome = "reject_loop"
                    break
            else:  # REJECT_DEAL
                pending, pending_by, incoming_offer = None, None, None
                incoming_text = (
                    f'{speaker.playerName} rejected your offer. They said: "{turn.talk or ""}"'
                )
            speaker, listener = listener, speaker
        else:
            result.outcome = "max_turns"
    finally:
        for agent in agents.values():
            agent.end_conversation()

    if result.outcome != "declined":
        print(f"  [NEGOTIATION] Outcome: {result.outcome}")
    return result
