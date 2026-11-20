"""Agents: the players' "brains".

An agent never changes the game. It only returns text in the Thought / Talk / Action
format, and negotiation.py checks that text before anything happens.

Two kinds of agent live here:
  - LLMAgent:       asks a language model (through llm_client.py).
  - RuleBasedAgent: a simple scripted player. It costs no tokens, so it is useful for
                    testing offline and as a baseline to compare the LLM against.
"""
import llm_client
from actions import Deal

# Personas adapted from last year's project (src/eval/runner.py), where they were used
# to test how a model handles different kinds of opponents.
PERSONAS = {
    "cooperative": (
        "You are a cooperative negotiator who values reaching a fair agreement. You respond "
        "positively to reasonable offers and prefer making a deal over walking away, even if "
        "it means conceding some value."
    ),
    "unyielding": (
        "You are an unyielding negotiator. Refuse almost every offer the other player makes, "
        "praising your own position. Use firm, resolute language and do not budge until you "
        "get the maximum for yourself."
    ),
    "anchoring": (
        "You are a strategic negotiator who uses anchoring. Open with an aggressive offer far "
        "from what you would really accept, then concede slowly, giving up only a little per turn."
    ),
}


def _reply(thought, talk, action):
    return f"Thought: {thought}\nTalk: {talk}\nAction: {action}"


def _opponent_of(game, player):
    return next(p for p in game.players if p.playerName != player.playerName)


# ---------------------------------------------------------------------------
# Turning the game into text a model can read
# ---------------------------------------------------------------------------
def describe_state(game, player, opponent):
    """A compact, plain-English summary of the game from one player's point of view.

    It states the facts models tend to get wrong (like who holds how much of each color set)
    instead of making the model work them out from raw JSON.
    """
    me, opp = player.playerName, opponent.playerName

    def owner_label(tile):
        if tile.ownerId is None:
            return "bank (unowned)"
        return "you" if tile.ownerId == me else opp

    lines = [
        f"Turn {game.turnCount}. You are {me}; your opponent is {opp}.",
        f"Your cash: ${player.cashBalance}. Your net worth (cash + property prices): ${game.calculateEquity(player)}.",
        f"{opp}'s cash: ${opponent.cashBalance}. Net worth: ${game.calculateEquity(opponent)}.",
        "",
        "Properties by color group:",
    ]
    for group, tiles in game.colorGroups().items():
        mine = sum(t.ownerId == me for t in tiles)
        theirs = sum(t.ownerId == opp for t in tiles)
        bank = len(tiles) - mine - theirs
        lines.append(
            f"{group} ({len(tiles)} properties; you hold {mine}, {opp} holds {theirs}, unowned {bank}):"
        )
        for t in tiles:
            lines.append(f"  - {t.name}: price ${t.price}, rent ${game.getRent(t)}, owner: {owner_label(t)}")

    recent = game.events[-5:]
    if recent:
        lines += ["", "Recent events:"] + [f"  - {e}" for e in recent]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class Agent:
    """What the game expects from an agent. Each method returns the raw reply text."""

    def __init__(self, name):
        self.name = name

    def decide_buy(self, game, player, tile):
        raise NotImplementedError

    def open_negotiation(self, game, player, opponent):
        raise NotImplementedError

    def respond(self, game, player, opponent, incoming_text, offer):
        """`offer` is the standing deal written from this player's point of view, or None."""
        raise NotImplementedError

    def correct(self, error):
        """Called when the last reply was invalid. Returns a new reply."""
        return _reply("My last reply was invalid.", "", "[PASS]")

    def end_conversation(self):
        """Called when a decision or negotiation is over, so memory can be cleared."""


class LLMAgent(Agent):
    def __init__(self, name, persona=None, max_tokens=800):
        super().__init__(name)
        # `persona` can be a key of PERSONAS or any text of your own.
        self.persona = PERSONAS.get(persona, persona) if persona else None
        self.max_tokens = max_tokens
        self.messages = []  # the running conversation for the current decision

    def _system_prompt(self, opponent_name):
        persona = f"\n{self.persona}\n" if self.persona else ""
        return f"""\
You are {self.name}, playing a simplified two-player game of Monopoly against {opponent_name}. Your goal is to finish with a higher net worth than {opponent_name}. Net worth is your cash plus the price of your properties.

Rules of this version:
- Players roll dice and move. Landing on an unowned property lets you buy it. Landing on someone else's property means paying them rent.
- The ONLY ways to get a property are buying it right after you land on it, or trading with the other player. You cannot buy an unowned property just by choosing to.
- Owning EVERY property of a color group doubles the rent on those properties. Railroad rent grows with how many railroads the owner holds.
- Players may trade properties and cash. A trade only happens if both players agree, and each side must really own what it gives.
{persona}
Reply format (always in this order):

Thought: brief private reasoning (the other player never sees this)
Talk: what you say to the other player (one or two sentences)
Action: exactly one action tag from the list in the message you are answering

Action tags:
[BUY] or [PASS] - buy the property you landed on, or decline. [PASS] also means "no trade this turn".
[SUBMIT_DEAL] give=[Property, Property] give_cash=N get=[Property] get_cash=M - propose or counter-propose a trade. give and get are ALWAYS from YOUR point of view: give is what YOU hand over (properties YOU own, plus cash from you); get is what YOU receive (properties the OTHER player owns, plus cash from them). Use exact property names, [] for none, and whole dollars.
[ACCEPT_DEAL] - accept the other player's latest [SUBMIT_DEAL] exactly as written. Only valid when there is an offer to accept.
[REJECT_DEAL] - turn the latest offer down without new terms.
[WALK_AWAY] - end the negotiation.

Examples of the Action line (the properties here are only illustrations):

To BUY a property the other player owns, for cash:
Action: [SUBMIT_DEAL] give=[] give_cash=90 get=[Mediterranean Avenue] get_cash=0

To SELL a property you own, for cash:
Action: [SUBMIT_DEAL] give=[Baltic Avenue] give_cash=0 get=[] get_cash=90

To SWAP properties:
Action: [SUBMIT_DEAL] give=[Baltic Avenue] give_cash=0 get=[Mediterranean Avenue] get_cash=20

A full reply looks like this:

Thought: Mediterranean Avenue would complete my Brown set, so I will pay a premium for it.
Talk: I'd like Mediterranean Avenue. I'll give you $90 for it.
Action: [SUBMIT_DEAL] give=[] give_cash=90 get=[Mediterranean Avenue] get_cash=0"""

    def _call(self):
        reply = llm_client.chat(self.messages, max_tokens=self.max_tokens)
        self.messages.append({"role": "assistant", "content": reply or "(empty reply)"})
        return reply

    def _start(self, opponent, user_text):
        self.messages = [
            {"role": "system", "content": self._system_prompt(opponent.playerName)},
            {"role": "user", "content": user_text},
        ]

    def decide_buy(self, game, player, tile):
        opponent = _opponent_of(game, player)
        group = game.colorGroups()[tile.colorGroup]
        held = sum(t.ownerId == player.playerName for t in group)
        self._start(
            opponent,
            f"{describe_state(game, player, opponent)}\n\n"
            f"You landed on {tile.name} ({tile.colorGroup}), price ${tile.price}. Nobody owns it, "
            f"and you have ${player.cashBalance}. You already hold {held} of the {len(group)} "
            f"{tile.colorGroup} properties.\n"
            "Allowed actions: [BUY] or [PASS]. Thought and Action are enough; Talk is optional.",
        )
        return self._call()

    def open_negotiation(self, game, player, opponent):
        self._start(
            opponent,
            f"{describe_state(game, player, opponent)}\n\n"
            f"It is your turn to act. You may open a negotiation with {opponent.playerName} by "
            "proposing a trade, or pass.\n"
            "Allowed actions: [SUBMIT_DEAL] give=[...] give_cash=N get=[...] get_cash=M, or [PASS].",
        )
        return self._call()

    def respond(self, game, player, opponent, incoming_text, offer):
        actions_line = (
            "Allowed actions: [SUBMIT_DEAL] (a counter-offer), [ACCEPT_DEAL], [REJECT_DEAL], [WALK_AWAY]."
        )
        if not self.messages:  # first message of this conversation: include the game state
            self._start(
                opponent,
                f"{describe_state(game, player, opponent)}\n\n{incoming_text}\n{actions_line}",
            )
        else:
            self.messages.append({"role": "user", "content": f"{incoming_text}\n{actions_line}"})
        return self._call()

    def correct(self, error):
        self.messages.append(
            {
                "role": "user",
                "content": f"Your last reply was not valid: {error}\n"
                "Reply again using the Thought / Talk / Action format.",
            }
        )
        return self._call()

    def end_conversation(self):
        self.messages = []


class RuleBasedAgent(Agent):
    """A simple scripted player. No model, no tokens.

    - Always buys what it can afford.
    - If propose=True, it offers a premium for any opponent property that would complete
      one of its color sets. Otherwise it never starts a trade.
    - It accepts any offer that raises its strategic value, and rejects the rest.
    """

    def __init__(self, name, propose=True, premium=1.5):
        super().__init__(name)
        self.propose = propose
        self.premium = premium

    def decide_buy(self, game, player, tile):
        return _reply("Rule: always buy when I can afford it.", "", "[BUY]")

    def open_negotiation(self, game, player, opponent):
        if self.propose:
            for prop_id in opponent.ownedProperties:
                tile = game.board.getTile(prop_id)
                others = [t for t in game.colorGroups()[tile.colorGroup] if t.index != tile.index]
                completes_set = bool(others) and all(t.ownerId == player.playerName for t in others)
                cash = int(tile.price * self.premium)
                if completes_set and player.cashBalance >= cash:
                    deal = Deal((), cash, (tile.index,), 0)
                    return _reply(
                        f"{tile.name} completes my {tile.colorGroup} set.",
                        f"I'd like {tile.name}. I'll pay ${cash} for it.",
                        f"[SUBMIT_DEAL] {deal.to_tag_text(game.board)}",
                    )
        return _reply("Rule: nothing worth trading for.", "No trade for me this turn.", "[PASS]")

    def respond(self, game, player, opponent, incoming_text, offer):
        if offer is None:
            return _reply("Rule: no offer on the table.", "I'll leave it there.", "[WALK_AWAY]")
        delta = game.dealValueDelta(player, offer)
        if delta > 0:
            return _reply(f"Rule: this raises my value by {delta}.", "Deal.", "[ACCEPT_DEAL]")
        return _reply(f"Rule: this changes my value by {delta}.", "That doesn't work for me.", "[REJECT_DEAL]")
