"""The decision loop: ask an agent for an action, check it, retry once, then act.

Used for:
  - ask_buy_decision: "buy or pass?"
  - run_negotiation:  the active player opens ONE negotiation, either
      * a targeted offer ([SUBMIT_DEAL] to=[Player]) followed by a short back-and-forth
        between those two players, or
      * an idea floated to everyone ([FLOAT_IDEA]): every other player replies with an
        offer or [IGNORE] without seeing the other replies (sealed), then the proposer
        accepts one, counters one (which continues as the two-player back-and-forth),
        or walks away.

The key rule: only a VALIDATED action ever changes the game. What an agent says in
"Talk" is just text the other agents read. This is what stops an agent from "trading"
a property it does not own.

The two-player protocol is borrowed from last year's project: players alternate turns,
a deal is proposed with [SUBMIT_DEAL], [ACCEPT_DEAL] only works on a standing offer,
and the loop ends on a deal, a walk-away, a repeated offer, or a turn cap.
"""
from dataclasses import asdict, dataclass, field

import llm_client
from actions import (
    BUY_ACTIONS,
    CHOOSE_ACTIONS,
    EVERYONE,
    IDEA_REPLY_ACTIONS,
    OPEN_ACTIONS,
    RESPOND_ACTIONS,
    ParsedAction,
    describe_side,
    parse_action,
    parse_reply,
)

MAX_ATTEMPTS = 2  # the first reply, plus one correction if it was invalid

ALLOWED = {
    "buy": BUY_ACTIONS,
    "open": OPEN_ACTIONS,
    "respond": RESPOND_ACTIONS,
    "idea_reply": IDEA_REPLY_ACTIONS,
    "choose": CHOOSE_ACTIONS,
}

# If a model still gives an invalid reply after the correction, do something harmless.
FALLBACK = {"buy": "PASS", "open": "PASS", "respond": "REJECT_DEAL",
            "idea_reply": "IGNORE", "choose": "WALK_AWAY"}


@dataclass
class Turn:
    """One agent decision, kept for the research log."""

    game_turn: int
    speaker: str
    decision: str  # "buy" | "open" | "respond" | "idea_reply" | "choose"
    kind: str  # the action actually taken, e.g. "SUBMIT_DEAL"
    thought: str | None = None
    talk: str | None = None
    deal: object = None
    deal_text: str | None = None
    target: str | None = None  # who a [SUBMIT_DEAL] was for / whose offer was accepted (engine-checked)
    format_ok: bool = True  # did the FIRST reply follow Thought / Talk / Action?
    attempts: int = 1
    fell_back: bool = False  # True if the replies were invalid and we used the harmless default
    error: str | None = None
    tokens: int = 0
    retry_errors: list = field(default_factory=list)  # why each earlier attempt was rejected
    first_reply: str | None = None  # the model's first reply, kept only if it had to be retried
    capped: bool = False  # True if the per-turn call budget stopped a call (attempts=0: never asked)
    finish_reasons: list = field(default_factory=list)  # per attempt: "stop", "length", ... (None = no model)
    prompt_tokens: int = 0  # tokens sent to the model for this decision
    completion_tokens: int = 0  # tokens the model produced for this decision

    def to_dict(self):
        return asdict(self)


def _empty_reply_error(finish_reason, allowed):
    """The message an agent gets back when its reply had no text."""
    if finish_reason == "length":
        return ("Your reply was cut off before you wrote anything: it ran out of room, probably while "
                "thinking. Think briefly, then write Thought (one short sentence), Talk, and "
                "'Action: [TAG]'. Allowed tags: " + ", ".join(f"[{a}]" for a in allowed) + ".")
    return ("Your reply was empty. Reply with Thought, Talk, then 'Action: [TAG]'. Allowed tags: "
            + ", ".join(f"[{a}]" for a in allowed) + ".")


@dataclass
class NegotiationResult:
    game_turn: int
    initiator: str
    responder: str | None  # the partner (chosen by the opening offer or the idea's pick); None if none
    # deal | walk_away | declined | max_turns | reject_loop | invalid_deal | call_cap | no_offers
    outcome: str = "max_turns"
    proposer: str | None = None
    deal_text: str | None = None  # from the proposer's point of view
    trade: dict | None = None  # value changes, filled in when a deal happens
    turns: list = field(default_factory=list)
    kind: str = "targeted"  # "targeted" | "broadcast"
    idea_text: str | None = None  # broadcast only: the idea, from the initiator's point of view
    offers: dict = field(default_factory=dict)  # broadcast only: {player: offer from their view}
    participants: list = field(default_factory=list)  # everyone who replied, in order of first reply

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Asking an agent for one valid action
# ---------------------------------------------------------------------------
def _check_choice(game, speaker, act, offers):
    """For the 'choose' decision: which offer is being accepted or countered.

    Returns (player, error). offers: {name: Deal from the speaker's point of view}.
    """
    named = act.source if act.kind == "ACCEPT_DEAL" else act.target
    key = "from" if act.kind == "ACCEPT_DEAL" else "to"
    options = ", ".join(offers)
    if named is None:
        if len(offers) == 1:
            return game.getPlayer(next(iter(offers))), None
        return None, f"Say which offer you mean: add {key}=[Player]. Offers came from: {options}."
    player = game.findPlayer(named)
    if player is None or player.playerName not in offers:
        return None, f"There is no offer from '{named}'. Offers came from: {options}."
    return player, None


def get_valid_action(game, agent, speaker, listener, decision, first_call,
                     pending=None, pending_by=None, offers=None):
    """Get a legal action from an agent. Retries once with the exact problem, then falls back.

    listener: the player already in this negotiation, or None (a buy decision, an opening
    move whose to=[...] still has to pick the partner, or choosing among offers).
    offers: for the 'choose' decision, {name: Deal from the speaker's point of view}.

    Every call (the first one and the retry) uses one unit of the game's per-turn call
    budget. If none is left, the agent is not asked and the safe default is used.
    """
    allowed = ALLOWED[decision]
    need_talk = decision != "buy"
    tokens_before = llm_client.total_tokens_used

    if not game.useAgentCall():
        return Turn(
            game_turn=game.turnCount,
            speaker=speaker.playerName,
            decision=decision,
            kind=FALLBACK[decision],
            attempts=0,
            capped=True,
            error="The call budget for this turn is used up, so the agent was not asked.",
        )

    prompt_before = llm_client.total_prompt_tokens
    completion_before = llm_client.total_completion_tokens
    finish_reasons = []

    def ask(call):
        """Run one agent call and note why the model's reply ended (None if no model was used)."""
        calls_before = llm_client.total_calls
        text = call()
        finish_reasons.append(llm_client.last_finish_reason if llm_client.total_calls > calls_before else None)
        return text

    reply = ask(first_call)
    attempts = 0
    first_format_ok = None
    first_reply = reply
    retry_errors = []
    capped = False
    while True:
        attempts += 1
        parsed = parse_reply(reply, need_talk)
        if not (reply or "").strip():
            act = ParsedAction(error=_empty_reply_error(finish_reasons[-1], allowed))
        else:
            act = parse_action(parsed, allowed, game.board)
        target = None
        if act.error is None:
            if decision == "choose" and act.kind in ("ACCEPT_DEAL", "SUBMIT_DEAL"):
                target, act.error = _check_choice(game, speaker, act, offers)
            if act.error is None and act.kind == "SUBMIT_DEAL":
                if target is None:
                    target, act.error = game.resolveTradeTarget(speaker, act.target, listener)
                if act.error is None:
                    act.error = game.validateTrade(speaker, target, act.deal.to_offer_data())
                if act.error is None:
                    act.error = game.lockConflict(speaker.playerName, target.playerName, act.deal.give_props)
            elif act.kind == "FLOAT_IDEA":
                act.error = game.validateIdea(speaker, act.deal)
            elif (act.kind == "ACCEPT_DEAL" and decision != "choose"
                  and (pending is None or pending_by == speaker.playerName)):
                act.error = "There is no offer from the other player to accept."
        if first_format_ok is None:
            first_format_ok = parsed.format_ok
        if act.error is None or attempts >= MAX_ATTEMPTS:
            break
        if not game.useAgentCall():
            capped = True  # invalid reply, but no budget left for the retry
            break
        retry_errors.append(act.error)
        reply = ask(lambda: agent.correct(act.error))

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
        target=target.playerName if (target and not fell_back) else None,
        format_ok=first_format_ok,
        attempts=attempts,
        fell_back=fell_back,
        error=act.error if fell_back else None,
        tokens=llm_client.total_tokens_used - tokens_before,
        retry_errors=retry_errors,
        first_reply=(first_reply or "")[:800] if retry_errors else None,
        capped=capped,
        finish_reasons=finish_reasons,
        prompt_tokens=llm_client.total_prompt_tokens - prompt_before,
        completion_tokens=llm_client.total_completion_tokens - completion_before,
    )


def _print_turn(game, turn):
    if game.showThoughts and turn.thought:
        print(f"     [thought] {turn.thought}")
    if turn.talk:
        print(f'     {turn.speaker}: "{turn.talk}"')
    text = f"[{turn.kind}]"
    if turn.decision == "idea_reply" and not turn.talk:
        text = f"{turn.speaker}: {text}"  # several players reply in a row; say who this was
    if turn.target:
        text += f" from {turn.target}" if turn.kind == "ACCEPT_DEAL" else f" to {turn.target}:"
    if turn.deal:
        text += f" {turn.deal.describe(game.board)}"
    note = ""
    if turn.capped and turn.attempts == 0:
        note = "   (call budget for this turn used up; the agent was not asked)"
    elif turn.capped:
        note = f"   (!) invalid, and no call budget left to retry. Problem: {turn.error}"
    elif turn.fell_back:
        note = f"   (!) invalid even after a retry, used the default. Problem: {turn.error}"
    elif turn.attempts > 1:
        note = f"   (retried once because: {turn.retry_errors[0]})"
    print(f"     -> {text}{note}")


# ---------------------------------------------------------------------------
# Buy decision
# ---------------------------------------------------------------------------
def ask_buy_decision(game, agent, player, tile):
    """Ask the agent whether to buy the property it landed on. Returns True to buy."""
    print(f"  -> [DECISION] {player.playerName} considers {tile.name} (${tile.price}).")
    try:
        # listener=None: a buy decision involves no other player ([BUY]/[PASS] only).
        turn = get_valid_action(
            game, agent, player, None, "buy", lambda: agent.decide_buy(game, player, tile)
        )
    finally:
        agent.end_conversation()
    _print_turn(game, turn)
    game.decisionLog.append(turn.to_dict())
    return turn.kind == "BUY"


# ---------------------------------------------------------------------------
# Messages forwarded between agents (Talk + engine-written terms; never the Thought)
# ---------------------------------------------------------------------------
def _offer_message(game, speaker, talk, deal_for_listener):
    """What the listener sees: the talk plus the deal, written from THEIR point of view."""
    give = describe_side(deal_for_listener.give_props, deal_for_listener.give_cash, game.board)
    get = describe_side(deal_for_listener.get_props, deal_for_listener.get_cash, game.board)
    return (
        f'{speaker.playerName} says: "{talk or ""}"\n'
        f"{speaker.playerName} proposes a deal. YOU would give: {give}. YOU would get: {get}."
    )


def _idea_message(game, proposer, talk, idea):
    """What every other player sees when an idea is floated. Wanted properties show their owner."""
    wanted = [f"{game.board.getTile(i).name} (owned by {game.board.getTile(i).ownerId})"
              for i in idea.get_props]
    if idea.get_cash:
        wanted.append(f"${idea.get_cash}")
    gives = describe_side(idea.give_props, idea.give_cash, game.board)
    name = proposer.playerName
    return (
        f'{name} floats an idea to every player: "{talk or ""}"\n'
        f"{name} wants: {' + '.join(wanted)}. {name} would give: {gives}.\n"
        f"You may reply with a concrete offer to {name}, or ignore it. Other players are "
        "replying too; you will not see their offers, and they will not see yours."
    )


def _choice_message(game, offers, ignored):
    """What the proposer sees after everyone replied: each offer from the proposer's view."""
    lines = ["Replies to your idea:"]
    for name, (deal, talk) in offers.items():
        give = describe_side(deal.give_props, deal.give_cash, game.board)
        get = describe_side(deal.get_props, deal.get_cash, game.board)
        lines.append(f'- {name} says: "{talk or ""}" Offer: YOU would give: {give}. YOU would get: {get}.')
    for name in ignored:
        lines.append(f"- {name} did not make an offer.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Negotiation
# ---------------------------------------------------------------------------
def _broadcast_round(game, result, proposer, idea_turn, agents):
    """Collect sealed replies to an idea, then let the proposer choose.

    Returns the proposer's counter-offer Turn (the negotiation continues with that player),
    or None when the negotiation is over (result.outcome is set).
    """
    idea = idea_turn.deal
    result.kind = "broadcast"
    result.idea_text = idea_turn.deal_text
    game.lockOffer(proposer.playerName, EVERYONE, idea.give_props)
    idea_text = _idea_message(game, proposer, idea_turn.talk, idea)

    offers = {}  # name -> (Deal from the PROPOSER's point of view, talk)
    ignored = []
    any_capped = False
    for player in game.tradePartners(proposer):  # turn order
        agent = game.agents[player.playerName]
        agents[player.playerName] = agent
        try:
            turn = get_valid_action(
                game, agent, player, proposer, "idea_reply",
                lambda: agent.respond_to_idea(game, player, proposer, idea_text, idea),
            )
        finally:
            agent.end_conversation()  # sealed: each reply is its own conversation
        _print_turn(game, turn)
        result.turns.append(turn.to_dict())
        any_capped = any_capped or turn.capped
        if turn.kind == "SUBMIT_DEAL":
            game.lockOffer(player.playerName, proposer.playerName, turn.deal.give_props)
            offers[player.playerName] = (turn.deal.flipped(), turn.talk)
            result.offers[player.playerName] = turn.deal_text
        else:
            ignored.append(player.playerName)

    if not offers:
        result.outcome = "call_cap" if any_capped else "no_offers"
        return None

    pagent = agents[proposer.playerName]
    turn = get_valid_action(
        game, pagent, proposer, None, "choose",
        lambda: pagent.choose_offer(game, proposer, _choice_message(game, offers, ignored),
                                    {n: d for n, (d, _) in offers.items()}),
        offers={n: d for n, (d, _) in offers.items()},
    )
    _print_turn(game, turn)
    result.turns.append(turn.to_dict())
    if turn.capped:
        result.outcome = "call_cap"
        return None
    if turn.kind == "WALK_AWAY":
        result.outcome = "walk_away"
        return None

    # A choice was made: every other offer, and the idea itself, is closed.
    game.releaseLocks()
    chosen = game.getPlayer(turn.target)
    result.responder = chosen.playerName
    if turn.kind == "ACCEPT_DEAL":
        their_offer = offers[chosen.playerName][0].flipped()  # back to the offerer's point of view
        trade = game.executeTrade(chosen, proposer, their_offer.to_offer_data())
        if trade:
            result.outcome = "deal"
            result.proposer = chosen.playerName
            result.deal_text = their_offer.to_tag_text(game.board)
            result.trade = trade
        else:
            result.outcome = "invalid_deal"
        return None
    return turn  # a counter-offer to `chosen`


def run_negotiation(game, initiator, responder=None, max_turns=6):
    """The initiator may open with a targeted offer or an idea for everyone.

    responder=None: the initiator's opening [SUBMIT_DEAL] picks the partner with to=[...].
    Passing a responder fixes the partner in advance (to=[...] may then be left out).
    A "turn" is one agent reply in the two-player back-and-forth (so max_turns=6 means at
    most 6 replies, plus retries). A floated idea, its replies and the proposer's choice
    together take the place of the opening reply; the per-turn call budget caps their cost.
    Returns a NegotiationResult.
    """
    result = NegotiationResult(
        game_turn=game.turnCount,
        initiator=initiator.playerName,
        responder=responder.playerName if responder else None,
    )
    agents = {p.playerName: game.agents[p.playerName] for p in (initiator, responder) if p}
    speaker, listener = initiator, responder
    pending = None  # the standing offer, written from the proposer's point of view
    pending_by = None
    incoming_text = None
    incoming_offer = None
    submits = {}  # each player's offers, to spot one being repeated

    # What the opening agent is told it may do: one fixed partner, or every possible partner.
    choices = [responder] if responder else game.tradePartners(initiator)
    print(f"  [NEGOTIATION] {initiator.playerName} may open a trade with "
          f"{', '.join(p.playerName for p in choices)}.")
    try:
        for index in range(max_turns):
            agent = agents[speaker.playerName]
            if index == 0:
                decision = "open"
                first_call = lambda: agent.open_negotiation(game, speaker, choices)
            else:
                decision = "respond"
                first_call = lambda: agent.respond(game, speaker, listener, incoming_text, incoming_offer)

            turn = get_valid_action(
                game, agent, speaker, listener, decision, first_call, pending, pending_by
            )
            _print_turn(game, turn)
            result.turns.append(turn.to_dict())
            kind = turn.kind
            if turn.capped:
                result.outcome = "call_cap"  # out of budget: end safely, nothing changes
                break

            if kind == "FLOAT_IDEA":
                turn = _broadcast_round(game, result, initiator, turn, agents)
                if turn is None:
                    break  # outcome already set
                kind = "SUBMIT_DEAL"  # the proposer countered one offer: continue as two players

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
                if listener is None:  # the opening offer (or the counter to an idea) chose the partner
                    listener = game.getPlayer(turn.target)
                    agents[listener.playerName] = game.agents[listener.playerName]
                    result.responder = listener.playerName
                game.lockOffer(speaker.playerName, listener.playerName, turn.deal.give_props)
                pending, pending_by = turn.deal, speaker.playerName
                incoming_offer = turn.deal.flipped()
                incoming_text = _offer_message(game, speaker, turn.talk, incoming_offer)
                history = submits.setdefault(speaker.playerName, [])
                history.append(turn.deal_text)
                if len(history) >= 3 and len(set(history[-3:])) == 1:
                    result.outcome = "reject_loop"
                    break
            else:  # REJECT_DEAL
                if pending_by:
                    game.releaseLocks(owner=pending_by)  # the rejected offer is closed
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
        game.releaseLocks()  # every offer closes when the negotiation ends, however it ends

    result.participants = list(dict.fromkeys(t["speaker"] for t in result.turns))
    if result.outcome != "declined":
        print(f"  [NEGOTIATION] Outcome: {result.outcome}")
    return result
