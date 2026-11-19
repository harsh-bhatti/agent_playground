"""Offline tests. No API key, no tokens, no internet needed.

Run from the project's top folder:

    python monopoly\\test_actions.py
"""
import contextlib
import io
import random

import llm_client
from actions import Deal, extract_section, parse_action, parse_deal, parse_reply, OPEN_ACTIONS, RESPOND_ACTIONS
from agents import Agent, LLMAgent, RuleBasedAgent
from board import Board
from gameManager import GameManager
from tile import PropertyTile

A, B = "Agent A", "Agent B"
BOARD = Board()


# ----------------------------------------------------------------- helpers
def reply(action, talk="ok", thought="thinking"):
    return f"Thought: {thought}\nTalk: {talk}\nAction: {action}"


class ScriptedAgent(Agent):
    """Returns canned replies in order, so we can test the game logic without a model."""

    def __init__(self, name, buy=(), open=(), respond=(), correct=(), idea=(), choose=()):
        super().__init__(name)
        self.q = {"buy": list(buy), "open": list(open), "respond": list(respond), "correct": list(correct),
                  "idea": list(idea), "choose": list(choose)}
        self.errors = []  # problems the game reported back to us
        self.incoming = []  # messages the game forwarded to us
        self.locks_seen = None  # game.locks at the moment this agent was asked to choose

    def respond_to_idea(self, game, player, proposer, idea_text, idea):
        self.incoming.append(idea_text)
        return self.q["idea"].pop(0) if self.q["idea"] else super().respond_to_idea(
            game, player, proposer, idea_text, idea)

    def choose_offer(self, game, player, offers_text, offers):
        self.incoming.append(offers_text)
        self.locks_seen = dict(game.locks)
        return self.q["choose"].pop(0)

    def decide_buy(self, game, player, tile):
        return self.q["buy"].pop(0)

    def open_negotiation(self, game, player, opponent):
        return self.q["open"].pop(0)

    def respond(self, game, player, opponent, incoming_text, offer):
        self.incoming.append(incoming_text)
        return self.q["respond"].pop(0)

    def correct(self, error):
        self.errors.append(error)
        return self.q["correct"].pop(0)


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def tile_id(name):
    return next(t.index for t in BOARD.tiles if isinstance(t, PropertyTile) and t.name == name)


def give_property(game, player_name, prop_name):
    tile = game.board.getTile(tile_id(prop_name))
    tile.ownerId = player_name
    game.getPlayer(player_name).addProperty(tile.index)


def make_game(agent_a=None, agent_b=None, max_turns=6):
    agents = {}
    if agent_a:
        agents[A] = agent_a
    if agent_b:
        agents[B] = agent_b
    game = GameManager([A, B], agents=agents, maxNegotiationTurns=max_turns)
    give_property(game, A, "Baltic Avenue")
    give_property(game, B, "Mediterranean Avenue")
    return game


def run(game):
    from negotiation import run_negotiation

    with quiet():
        return run_negotiation(game, game.getPlayer(A), game.getPlayer(B), game.maxNegotiationTurns)


def check_invariants(game):
    """Every property has exactly one owner, and the books agree with the tiles."""
    seen = {}
    for p in game.players:
        assert len(p.ownedProperties) == len(set(p.ownedProperties)), "duplicate in one player's list"
        assert p.cashBalance >= 0, "negative cash"
        for pid in p.ownedProperties:
            assert pid not in seen, f"property {pid} owned twice"
            seen[pid] = p.playerName
            assert game.board.getTile(pid).ownerId == p.playerName, "tile owner disagrees with player"
    for t in game.board.tiles:
        if isinstance(t, PropertyTile) and t.ownerId is not None:
            assert seen.get(t.index) == t.ownerId, "tile has an owner the player does not list"


OFFER = "[SUBMIT_DEAL] give=[] give_cash=90 get=[Mediterranean Avenue] get_cash=0"


# ------------------------------------------------------------ parsing tests
def test_parse_reply():
    p = parse_reply("Thought: hmm\nTalk: hello there\nAction: [PASS]")
    assert (p.thought, p.talk, p.action, p.format_ok) == ("hmm", "hello there", "[PASS]", True)


def test_empty_talk_does_not_swallow_action():
    p = parse_reply("Thought: x\nTalk: \nAction: [BUY]")
    assert p.talk == "" and p.action == "[BUY]" and p.format_ok
    assert extract_section("Talk:\nAction: [BUY]", "Talk") == ""


def test_buy_reply_without_talk_is_ok():
    assert parse_reply("Thought: x\nAction: [BUY]", need_talk=False).format_ok
    assert not parse_reply("Thought: x\nAction: [BUY]", need_talk=True).format_ok


def test_markdown_and_think_blocks_are_cleaned():
    raw = "<think>secret</think>**Thought:** a\n**Talk:** b\n**Action:** `[PASS]`"
    p = parse_reply(raw)
    assert p.action == "[PASS]" and p.talk == "b" and "secret" not in p.raw


def test_wrong_order_is_flagged():
    assert not parse_reply("Action: [PASS]\nThought: x\nTalk: y").format_ok


def test_parse_deal():
    deal, err = parse_deal("[SUBMIT_DEAL] give=[Reading Railroad] give_cash=$50 get=[Baltic] get_cash=0", BOARD)
    assert err is None
    assert deal.give_props == (tile_id("Reading Railroad"),)
    assert deal.give_cash == 50
    assert deal.get_props == (tile_id("Baltic Avenue"),)  # partial names are matched


def test_parse_deal_errors():
    _, err = parse_deal("[SUBMIT_DEAL] give=[Boardwalk] give_cash=0 get=[] get_cash=0", BOARD)
    assert "Unknown property" in err
    _, err = parse_deal("[SUBMIT_DEAL] give=[] give_cash=0 get=[] get_cash=0", BOARD)
    assert "empty" in err
    _, err = parse_deal("[SUBMIT_DEAL] please", BOARD)
    assert "Could not read" in err
    _, err = parse_deal("[SUBMIT_DEAL] give=[Avenue] give_cash=0 get=[] get_cash=0", BOARD)
    assert "Unknown property" in err  # "Avenue" is ambiguous


def test_deal_flip():
    d = Deal((1,), 10, (3,), 20)
    assert d.flipped() == Deal((3,), 20, (1,), 10)
    assert d.flipped().flipped() == d


def test_parse_action_rules():
    def act(text, allowed):
        return parse_action(parse_reply(text), allowed, BOARD)

    assert act(reply("[PASS]"), OPEN_ACTIONS).kind == "PASS"
    assert "not allowed" in act(reply("[ACCEPT_DEAL]"), OPEN_ACTIONS).error
    assert "more than one" in act(reply("[PASS] or [SUBMIT_DEAL]"), OPEN_ACTIONS).error
    assert "No action tag" in act(reply("I think I will pass"), OPEN_ACTIONS).error
    assert act("no labels at all, [PASS]", OPEN_ACTIONS).kind == "PASS"  # forgiving fallback
    assert act(reply(OFFER), RESPOND_ACTIONS).kind == "SUBMIT_DEAL"


# ---------------------------------------------------------- game rule tests
def test_rent_rules():
    game = make_game()
    baltic = game.board.getTile(tile_id("Baltic Avenue"))
    assert game.getRent(baltic) == 4  # alone: normal rent
    give_property(game, A, "Mediterranean Avenue")
    game.getPlayer(B).removeProperty(tile_id("Mediterranean Avenue"))
    assert game.getRent(baltic) == 8  # full Brown set: doubled
    give_property(game, B, "Reading Railroad")
    reading = game.board.getTile(tile_id("Reading Railroad"))
    assert game.getRent(reading) == 25
    give_property(game, B, "Pennsylvania Railroad")
    assert game.getRent(reading) == 50  # two railroads


def test_validation_blocks_fake_properties():
    """The old bug: trading a property you do not own made it appear out of thin air."""
    game = make_game()
    a, b = game.getPlayer(A), game.getPlayer(B)
    fake = {"giveCash": 0, "giveProps": [tile_id("Reading Railroad")], "receiveCash": 0, "receiveProps": []}
    assert "does not own" in game.validateTrade(a, b, fake)
    with quiet():
        assert game.executeTrade(a, b, fake) is None
    check_invariants(game)
    assert b.ownedProperties == [tile_id("Mediterranean Avenue")]


def test_error_messages_explain_give_vs_get():
    """The live run showed models mixing up give/get, so the error must say how to fix it."""
    game = make_game()
    a, b = game.getPlayer(A), game.getPlayer(B)
    # A tries to "receive" its OWN property (it meant to sell it)
    wrong_get = {"giveCash": 0, "giveProps": [], "receiveCash": 50, "receiveProps": [tile_id("Baltic Avenue")]}
    msg = game.validateTrade(a, b, wrong_get)
    assert "does not own" in msg and "SELL" in msg and "give=[...]" in msg
    # A tries to "give" the OTHER player's property (it meant to buy it)
    wrong_give = {"giveCash": 0, "giveProps": [tile_id("Mediterranean Avenue")], "receiveCash": 0, "receiveProps": []}
    msg = game.validateTrade(a, b, wrong_give)
    assert "does not own" in msg and "BUY" in msg and "get=[...]" in msg


def test_buy_outside_a_landing_is_explained():
    act = parse_action(parse_reply(reply("[BUY]")), OPEN_ACTIONS, BOARD)
    assert "right after landing" in act.error and "SUBMIT_DEAL" in act.error


def test_validation_blocks_overspending():
    game = make_game()
    a, b = game.getPlayer(A), game.getPlayer(B)
    offer = {"giveCash": 99999, "giveProps": [], "receiveCash": 0, "receiveProps": [tile_id("Mediterranean Avenue")]}
    assert "only has" in game.validateTrade(a, b, offer)


# -------------------------------------------------------- negotiation tests
def test_negotiation_deal():
    a = ScriptedAgent(A, open=[reply(OFFER, "I'll pay $90")])
    b = ScriptedAgent(B, respond=[reply("[ACCEPT_DEAL]", "Deal")])
    game = make_game(a, b)
    result = run(game)
    assert result.outcome == "deal" and result.proposer == A
    assert tile_id("Mediterranean Avenue") in game.getPlayer(A).ownedProperties
    assert game.getPlayer(A).cashBalance == 1410 and game.getPlayer(B).cashBalance == 1590
    check_invariants(game)


def test_private_thought_is_not_forwarded():
    a = ScriptedAgent(A, open=[reply(OFFER, "public words", thought="TOP SECRET PLAN")])
    b = ScriptedAgent(B, respond=[reply("[WALK_AWAY]")])
    run(make_game(a, b))
    assert "public words" in b.incoming[0]
    assert "TOP SECRET" not in b.incoming[0]
    assert "YOU would give: Mediterranean Avenue" in b.incoming[0]  # flipped to B's point of view
    assert "YOU would get: $90" in b.incoming[0]


def test_counter_offer_then_accept():
    counter = "[SUBMIT_DEAL] give=[Mediterranean Avenue] give_cash=0 get=[] get_cash=150"
    a = ScriptedAgent(A, open=[reply(OFFER)], respond=[reply("[ACCEPT_DEAL]")])
    b = ScriptedAgent(B, respond=[reply(counter, "make it $150")])
    game = make_game(a, b)
    result = run(game)
    assert result.outcome == "deal" and result.proposer == B
    assert game.getPlayer(A).cashBalance == 1350  # paid B's price of $150, not the original $90
    check_invariants(game)


def test_invalid_deal_is_corrected():
    fake = "[SUBMIT_DEAL] give=[Reading Railroad] give_cash=0 get=[] get_cash=50"  # A does not own it
    a = ScriptedAgent(A, open=[reply(fake)], correct=[reply(OFFER)])
    b = ScriptedAgent(B, respond=[reply("[ACCEPT_DEAL]")])
    game = make_game(a, b)
    result = run(game)
    assert result.outcome == "deal"
    assert "does not own" in a.errors[0]  # the exact problem was sent back
    assert result.turns[0]["attempts"] == 2
    assert "does not own" in result.turns[0]["retry_errors"][0]
    assert "Reading Railroad" in result.turns[0]["first_reply"]
    check_invariants(game)


def test_invalid_twice_falls_back_safely():
    fake = "[SUBMIT_DEAL] give=[Reading Railroad] give_cash=0 get=[] get_cash=50"
    a = ScriptedAgent(A, open=[reply(fake)], correct=[reply(fake)])
    game = make_game(a, ScriptedAgent(B))
    result = run(game)
    assert result.outcome == "declined" and result.turns[0]["fell_back"]
    check_invariants(game)


def test_accept_with_no_offer_is_refused():
    a = ScriptedAgent(A, open=[reply(OFFER)], respond=[reply("[ACCEPT_DEAL]")], correct=[reply("[WALK_AWAY]")])
    b = ScriptedAgent(B, respond=[reply("[REJECT_DEAL]")])
    result = run(make_game(a, b))
    assert result.outcome == "walk_away"
    assert "no offer" in a.errors[0]


def test_walk_away_and_decline():
    a = ScriptedAgent(A, open=[reply(OFFER)])
    b = ScriptedAgent(B, respond=[reply("[WALK_AWAY]")])
    assert run(make_game(a, b)).outcome == "walk_away"
    a = ScriptedAgent(A, open=[reply("[PASS]")])
    assert run(make_game(a, ScriptedAgent(B))).outcome == "declined"


def test_max_turns_cap():
    def offer(n):
        return reply(f"[SUBMIT_DEAL] give=[] give_cash={n} get=[Mediterranean Avenue] get_cash=0")

    def counter(n):
        return reply(f"[SUBMIT_DEAL] give=[Mediterranean Avenue] give_cash=0 get=[] get_cash={n}")

    a = ScriptedAgent(A, open=[offer(10)], respond=[offer(20), offer(30)])
    b = ScriptedAgent(B, respond=[counter(500), counter(400), counter(300)])
    assert run(make_game(a, b, max_turns=4)).outcome == "max_turns"


def test_repeated_offer_loop_is_stopped():
    a = ScriptedAgent(A, open=[reply(OFFER)], respond=[reply(OFFER), reply(OFFER)])
    b = ScriptedAgent(B, respond=[reply("[REJECT_DEAL]"), reply("[REJECT_DEAL]")])
    assert run(make_game(a, b, max_turns=10)).outcome == "reject_loop"


def test_rule_based_agents_complete_a_set():
    game = make_game(RuleBasedAgent(A), RuleBasedAgent(B))
    result = run(game)
    assert result.outcome == "deal"
    assert game.ownsFullSet(A, "Brown")
    check_invariants(game)


# ------------------------------------------------------- LLMAgent (mocked)
def test_llm_agent_builds_conversation():
    sent = []

    def fake_chat(messages, max_tokens=800, temperature=None):
        sent.append([dict(m) for m in messages])  # copy: the list grows after the call
        return reply("[PASS]")

    real = llm_client.chat
    llm_client.chat = fake_chat
    try:
        agent = LLMAgent(A, persona="cooperative")
        game = make_game(agent, ScriptedAgent(B))
        a, b = game.getPlayer(A), game.getPlayer(B)

        agent.open_negotiation(game, a, b)
        assert len(sent[0]) == 2  # system + user
        assert "Properties by color group" in sent[0][1]["content"]
        assert "cooperative negotiator" in sent[0][0]["content"]
        assert "Agent B" in sent[0][0]["content"]

        agent.respond(game, a, b, f"{B} rejected your offer.", None)
        # Conversation memory is kept, and only a short message is added (no second state dump).
        assert len(sent[1]) == 4
        assert "Properties by color group" not in sent[1][3]["content"]

        agent.correct("example problem")
        assert len(sent[2]) == 6 and "example problem" in sent[2][5]["content"]

        agent.end_conversation()
        assert agent.messages == []

        tile = game.board.getTile(tile_id("Vermont Avenue"))
        agent.decide_buy(game, a, tile)
        assert "[BUY] or [PASS]" in sent[3][1]["content"] and len(sent[3]) == 2
    finally:
        llm_client.chat = real


# ------------------------------------------- llm_client (fake client, no API)
def test_createai_provider_settings():
    assert llm_client.BASE_URL.endswith("/v1")  # the OpenAI-compatible API lives under /v1
    assert llm_client.API_KEY_ENV == "CREATEAI_API_KEY"
    assert "groq" not in llm_client.BASE_URL


class _FakeCompletions:
    """Stands in for client.chat.completions. Raises the queued errors, then replies."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = []

    def create(self, **kwargs):
        import types

        self.calls.append(dict(kwargs))
        if self.errors:
            raise self.errors.pop(0)
        message = types.SimpleNamespace(content="hello")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
            usage=types.SimpleNamespace(total_tokens=7),
        )


def _bad_request(text):
    import httpx2
    from openai import BadRequestError

    response = httpx2.Response(400, request=httpx2.Request("POST", "http://offline.test"))
    return BadRequestError(text, response=response, body=None)


def _run_with_completions(completions, func):
    """Run func with llm_client talking to a fake `completions` object; restore everything after."""
    import types

    saved = {k: getattr(llm_client, k) for k in (
        "_client", "_use_max_completion_tokens", "_skip_temperature", "total_calls", "total_tokens_used",
        "total_prompt_tokens", "total_completion_tokens", "empty_replies", "cut_off_replies")}
    llm_client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    llm_client._use_max_completion_tokens = llm_client._skip_temperature = False
    try:
        with quiet():
            result = func()
    finally:
        for k, v in saved.items():
            setattr(llm_client, k, v)
    return result, completions


def _with_fake_client(errors, func):
    result, completions = _run_with_completions(_FakeCompletions(errors), func)
    return result, completions.calls


class _SeqCompletions:
    """Fake client.chat.completions that returns (content, finish_reason) pairs in order."""

    def __init__(self, replies, prompt_tokens=100, completion_tokens=50):
        self.replies = list(replies)
        self.calls = []
        self.usage = (prompt_tokens, completion_tokens)

    def create(self, **kwargs):
        import types

        self.calls.append(dict(kwargs))
        content, finish = self.replies.pop(0)
        message = types.SimpleNamespace(content=content)
        p, c = self.usage
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason=finish)],
            usage=types.SimpleNamespace(total_tokens=p + c, prompt_tokens=p, completion_tokens=c),
        )


def test_chat_adapts_when_model_rejects_options():
    errors = [_bad_request("Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."),
              _bad_request("Unsupported value: 'temperature' does not support 0.7.")]
    reply_text, calls = _with_fake_client(errors, lambda: llm_client.chat([{"role": "user", "content": "hi"}]))
    assert reply_text == "hello" and len(calls) == 3
    assert "max_tokens" not in calls[2] and calls[2]["max_completion_tokens"] == 800
    assert "temperature" not in calls[2]


def test_chat_reports_other_bad_requests():
    try:
        _with_fake_client([_bad_request("model not enabled for this project")],
                          lambda: llm_client.chat([{"role": "user", "content": "hi"}]))
        raise AssertionError("expected LLMError")
    except llm_client.LLMError as error:
        assert "model not enabled" in str(error)


# -------------------------------------------------------- full random games
def test_full_mock_games_keep_ownership_consistent():
    for seed in range(20):
        random.seed(seed)
        game = GameManager([A, B], agents={A: RuleBasedAgent(A), B: RuleBasedAgent(B)})
        with quiet():
            for _ in range(150):
                game.executeTurn()
                check_invariants(game)


# ------------------------------------------------- bankruptcy and game end
INCOME_TAX = 4  # board index of Income Tax ($200)


def land_on(game, player_name, index):
    """Put a player on Go and move them to `index`, as if they rolled that number."""
    player = game.getPlayer(player_name)
    player.boardPosition = 0
    with quiet():
        game.resolveMovement(player, index)
    return player


def test_tax_is_paid_when_affordable():
    game = make_game()
    a = land_on(game, A, INCOME_TAX)
    assert a.cashBalance == 1300 and not a.isBankrupt and not game.isGameOver()
    assert any("paid $200 Income Tax to the bank" in e for e in game.events)


def test_unaffordable_tax_bankrupts_and_returns_properties_to_bank():
    game = make_game()
    game.getPlayer(A).cashBalance = 50
    a = land_on(game, A, INCOME_TAX)
    assert a.isBankrupt and a.cashBalance == 0 and a.ownedProperties == []
    assert game.board.getTile(tile_id("Baltic Avenue")).ownerId is None  # back to the bank
    assert game.getPlayer(B).cashBalance == 1500  # the bank got the $50, not B
    assert game.winner == B and game.endReason == "bankruptcy"
    assert not any("paid $200" in e for e in game.events)  # the old bug: "paid" with no payment
    check_invariants(game)


def test_unaffordable_rent_gives_creditor_cash_and_properties():
    game = make_game()
    game.getPlayer(A).cashBalance = 1  # Mediterranean rent is $2
    a = land_on(game, A, tile_id("Mediterranean Avenue"))
    b = game.getPlayer(B)
    assert a.isBankrupt and a.cashBalance == 0
    assert b.cashBalance == 1501
    assert tile_id("Baltic Avenue") in b.ownedProperties
    assert game.board.getTile(tile_id("Baltic Avenue")).ownerId == B
    assert game.ownsFullSet(B, "Brown")
    record = game.bankruptcies[0]
    assert (record["player"], record["owed"], record["creditor"], record["cash_handed_over"]) == (A, 2, B, 1)
    assert game.winner == B
    check_invariants(game)


def test_exact_payment_is_not_bankruptcy():
    game = make_game()
    game.getPlayer(A).cashBalance = 200
    a = land_on(game, A, INCOME_TAX)
    assert a.cashBalance == 0 and not a.isBankrupt and not game.isGameOver()


def test_no_turns_after_game_over():
    game = make_game()
    game.getPlayer(A).cashBalance = 0
    land_on(game, A, INCOME_TAX)
    assert game.isGameOver()
    turns, cash = game.turnCount, game.getPlayer(B).cashBalance
    with quiet():
        for _ in range(5):
            game.executeTurn()
    assert game.turnCount == turns and game.getPlayer(B).cashBalance == cash
    game.finishByTurnLimit()  # must not overwrite the bankruptcy result
    assert game.endReason == "bankruptcy" and game.winner == B


def test_bankrupt_on_doubles_does_not_roll_again_and_is_skipped():
    # Three players without agents, so the game continues after one is eliminated.
    game = GameManager([A, B, "Agent C"])
    game.getPlayer(A).cashBalance = 10
    game.rollDice = lambda: (INCOME_TAX, True)  # A rolls doubles onto Income Tax
    with quiet():
        game.executeTurn()
    assert game.getPlayer(A).isBankrupt and not game.isGameOver()
    assert game.players[game.activePlayerIndex].playerName == B  # A did not roll again
    game.rollDice = lambda: (2, False)
    order = []
    with quiet():
        for _ in range(3):
            order.append(game.players[game.activePlayerIndex].playerName)
            game.executeTurn()
    assert order == [B, "Agent C", B]  # A is skipped


def test_turn_limit_winner_is_highest_net_worth():
    game = make_game()
    game.getPlayer(A).cashBalance += 10
    with quiet():
        game.finishByTurnLimit()
    assert game.winner == A and game.endReason == "turn_limit"


def test_turn_limit_tie_is_a_draw():
    game = make_game()  # both have $1500 + one $60 property
    with quiet():
        game.finishByTurnLimit()
    assert game.winner is None and game.endReason == "turn_limit"


def test_cash_transfer_never_fails_silently():
    game = make_game()
    a, b = game.getPlayer(A), game.getPlayer(B)
    try:
        game._transferCash(a, b, 99999)
        raise AssertionError("expected an error")
    except RuntimeError:
        pass
    assert a.cashBalance == 1500 and b.cashBalance == 1500  # nothing moved


def test_trade_with_bankrupt_player_is_refused():
    game = make_game()
    a, b = game.getPlayer(A), game.getPlayer(B)
    b.isBankrupt = True
    offer = {"giveCash": 10, "giveProps": [], "receiveCash": 0, "receiveProps": []}
    assert "bankrupt" in game.validateTrade(a, b, offer)


def test_result_is_in_summary_and_log_data():
    import main

    game = make_game()
    game.getPlayer(A).cashBalance = 0
    land_on(game, A, INCOME_TAX)
    result = game.gameResult()
    assert result["winner"] == B and result["end_reason"] == "bankruptcy"
    assert result["bankruptcies"][0]["player"] == A
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main.print_summary(game, stopped_early=False)
    assert f"Winner: {B} ({A} went bankrupt on turn" in out.getvalue()
    assert f"{A} (BANKRUPT)" in out.getvalue()


def test_full_mock_games_end_with_a_consistent_result():
    """Low starting cash makes bankruptcies likely. The books must stay consistent throughout.

    Not every game goes bankrupt: on this small board, $200 for passing Go outpaces rent and
    tax, so a player who survives the first laps usually never goes broke.
    """
    bankrupt_games = 0
    for seed in range(20):
        random.seed(seed)
        game = GameManager([A, B], agents={A: RuleBasedAgent(A), B: RuleBasedAgent(B)})
        for p in game.players:
            p.cashBalance = 150
        with quiet():
            for _ in range(300):
                game.executeTurn()
                check_invariants(game)
                if game.isGameOver():
                    break
            game.finishByTurnLimit()
        assert game.isGameOver()
        if game.endReason == "bankruptcy":
            bankrupt_games += 1
            survivors = game.activePlayers()
            assert len(survivors) == 1 and survivors[0].playerName == game.winner
            assert len(game.bankruptcies) == 1
        else:
            assert game.endReason == "turn_limit" and not game.bankruptcies
    assert bankrupt_games >= 5, f"only {bankrupt_games} of 20 games had a bankruptcy"


# ----------------------------------------------------- N-player (phase 1)
C, D = "Agent C", "Agent D"
FOUR = [A, B, C, D]


def turn_order(game, turns):
    """Play `turns` turns (no doubles, 2 steps each) and return who rolled each time."""
    game.rollDice = lambda: (2, False)
    order = []
    with quiet():
        for _ in range(turns):
            order.append(game.players[game.activePlayerIndex].playerName)
            game.executeTurn()
    return order


def test_player_names_and_agents_for_four_players():
    import argparse
    import main

    names = main.player_names(4)
    assert names == FOUR
    args = argparse.Namespace(mode="mock", persona_a=None, persona_b=None)
    agents = main.build_agents(args, names)
    assert list(agents) == FOUR and all(isinstance(a, RuleBasedAgent) for a in agents.values())
    args.mode = "llm-vs-rule"  # builds objects only; nothing calls the model
    agents = main.build_agents(args, names)
    assert isinstance(agents[A], LLMAgent)
    assert all(isinstance(agents[n], RuleBasedAgent) and not agents[n].propose for n in (B, C, D))


def test_four_player_turn_order():
    assert turn_order(GameManager(FOUR), 5) == [A, B, C, D, A]


def test_four_player_bankrupt_player_is_skipped():
    game = GameManager(FOUR)
    with quiet():
        game.declareBankruptcy(game.getPlayer(B), 100, None, "test")
    assert not game.isGameOver()
    assert turn_order(game, 4) == [A, C, D, A]


def test_four_player_bankruptcy_pays_the_right_creditor():
    game = GameManager(FOUR)
    give_property(game, C, "Mediterranean Avenue")
    give_property(game, A, "Baltic Avenue")
    game.getPlayer(A).cashBalance = 1  # rent is $2
    a = land_on(game, A, tile_id("Mediterranean Avenue"))
    c = game.getPlayer(C)
    assert a.isBankrupt and c.cashBalance == 1501 and tile_id("Baltic Avenue") in c.ownedProperties
    assert game.getPlayer(B).cashBalance == 1500 and game.getPlayer(D).cashBalance == 1500
    assert not game.isGameOver()  # three players are still in
    check_invariants(game)


def test_four_player_last_player_standing():
    game = GameManager(FOUR)
    with quiet():
        for name in (A, C):
            game.declareBankruptcy(game.getPlayer(name), 100, None, "test")
        assert not game.isGameOver()
        game.declareBankruptcy(game.getPlayer(D), 100, B, "rent")
    assert game.isGameOver() and game.winner == B and game.endReason == "bankruptcy"
    assert [b["player"] for b in game.bankruptcies] == [A, C, D]


def test_four_player_buy_decision_uses_the_agent():
    agent = ScriptedAgent(A, buy=[reply("[BUY]")])
    game = GameManager(FOUR, agents={A: agent})
    a = land_on(game, A, tile_id("Vermont Avenue"))
    assert tile_id("Vermont Avenue") in a.ownedProperties and a.cashBalance == 1400
    assert game.decisionLog[0]["kind"] == "BUY"


def test_describe_state_lists_every_opponent():
    from agents import describe_state

    game = GameManager(FOUR)
    give_property(game, C, "Reading Railroad")
    with quiet():
        game.declareBankruptcy(game.getPlayer(D), 100, None, "test")
    text = describe_state(game, game.getPlayer(A))
    assert "Your opponents: Agent B, Agent C." in text
    assert "Out of the game (bankrupt): Agent D." in text
    assert "Agent C: cash $1500" in text and "properties: Reading Railroad" in text
    assert "Agent C holds 1" in text and "owner: Agent C" in text
    assert "Agent D: cash" not in text  # bankrupt players are not listed as opponents


def test_llm_prompt_names_all_opponents():
    sent = []
    real = llm_client.chat
    llm_client.chat = lambda messages, **kw: sent.append([dict(m) for m in messages]) or reply("[PASS]")
    try:
        agent = LLMAgent(A)
        game = GameManager(FOUR, agents={A: agent})
        agent.decide_buy(game, game.getPlayer(A), game.board.getTile(tile_id("Vermont Avenue")))
    finally:
        llm_client.chat = real
    system = sent[0][0]["content"]
    assert "with 4 players" in system and "Agent B, Agent C, Agent D" in system
    assert "two-player" not in system and "highest net worth of all players" in system


# --------------------------------------------- choosing a partner (phase 2)
def make_four(**scripted):
    """4-player game: A owns Baltic, B owns Reading Railroad, C owns Mediterranean.

    Pass ScriptedAgents by player name (a=..., c=...); everyone else gets a do-nothing agent.
    """
    agents = {n: scripted.get(n[-1].lower()) or ScriptedAgent(n) for n in FOUR}
    game = GameManager(FOUR, agents=agents)
    give_property(game, A, "Baltic Avenue")
    give_property(game, B, "Reading Railroad")
    give_property(game, C, "Mediterranean Avenue")
    return game


def open_four(game):
    from negotiation import run_negotiation

    with quiet():
        return run_negotiation(game, game.getPlayer(A), None, game.maxNegotiationTurns)


TO_C = "[SUBMIT_DEAL] to=[Agent C] give=[] give_cash=90 get=[Mediterranean Avenue] get_cash=0"


def test_parse_to_target():
    def target(action):
        return parse_action(parse_reply(reply(action)), OPEN_ACTIONS, BOARD).target

    assert target(TO_C) == "Agent C"
    assert target("[SUBMIT_DEAL] to=Agent C give=[] give_cash=90 get=[Baltic] get_cash=0") == "Agent C"
    assert target("[SUBMIT_DEAL] give=[] give_cash=9 get=[Baltic] get_cash=0 to=[Agent D]") == "Agent D"
    assert target(OFFER) is None


def test_resolve_trade_target_rules():
    game = make_four()
    a = game.getPlayer(A)
    with quiet():
        game.declareBankruptcy(game.getPlayer(D), 10, None, "test")
    resolve = game.resolveTradeTarget
    assert resolve(a, "Agent C")[0].playerName == C
    assert resolve(a, "agent c")[0].playerName == C and resolve(a, "C")[0].playerName == C
    assert "to=[Player]" in resolve(a, None)[1] and "Agent B, Agent C" in resolve(a, None)[1]
    assert "no player called" in resolve(a, "Agent Z")[1]
    assert "yourself" in resolve(a, "Agent A")[1]
    assert "bankrupt" in resolve(a, "Agent D")[1]
    # inside a negotiation with B: to= may be left out, but may not name someone else
    assert resolve(a, None, game.getPlayer(B))[0].playerName == B
    assert "negotiating with Agent B" in resolve(a, "Agent C", game.getPlayer(B))[1]


def test_target_without_agent_is_refused():
    game = GameManager(FOUR, agents={A: ScriptedAgent(A), B: ScriptedAgent(B)})
    _, err = game.resolveTradeTarget(game.getPlayer(A), "Agent C")
    assert "cannot negotiate" in err
    assert game.resolveTradeTarget(game.getPlayer(A), None)[0].playerName == B  # only one partner


def test_validation_names_the_real_owner_and_unowned_properties():
    game = make_four()
    a, b = game.getPlayer(A), game.getPlayer(B)
    ask_b_for_c = {"giveCash": 90, "giveProps": [], "receiveCash": 0,
                   "receiveProps": [tile_id("Mediterranean Avenue")]}
    msg = game.validateTrade(a, b, ask_b_for_c)
    assert "belongs to Agent C, not Agent B" in msg and "to=[Agent C]" in msg
    ask_unowned = {"giveCash": 90, "giveProps": [], "receiveCash": 0,
                   "receiveProps": [tile_id("Vermont Avenue")]}
    assert "unowned" in game.validateTrade(a, b, ask_unowned)


def test_four_player_targeted_deal():
    a = ScriptedAgent(A, open=[reply(TO_C, "I'll pay $90")])
    c = ScriptedAgent(C, respond=[reply("[ACCEPT_DEAL]", "Deal")])
    game = make_four(a=a, c=c)
    result = open_four(game)
    assert result.outcome == "deal" and result.responder == C and result.turns[0]["target"] == C
    assert game.ownsFullSet(A, "Brown")
    assert game.getPlayer(A).cashBalance == 1410 and game.getPlayer(C).cashBalance == 1590
    assert game.getPlayer(B).cashBalance == 1500 and game.getPlayer(D).cashBalance == 1500  # untouched
    assert "YOU would give: Mediterranean Avenue" in c.incoming[0]
    check_invariants(game)


def test_missing_target_is_corrected_with_a_retry():
    a = ScriptedAgent(A, open=[reply(OFFER)], correct=[reply(TO_C)])
    c = ScriptedAgent(C, respond=[reply("[ACCEPT_DEAL]")])
    game = make_four(a=a, c=c)
    result = open_four(game)
    assert result.outcome == "deal" and "to=[Player]" in a.errors[0]
    assert result.turns[0]["attempts"] == 2


def test_offer_to_the_wrong_owner_falls_back_safely():
    to_b = "[SUBMIT_DEAL] to=[Agent B] give=[] give_cash=90 get=[Mediterranean Avenue] get_cash=0"
    a = ScriptedAgent(A, open=[reply(to_b)], correct=[reply(to_b)])
    game = make_four(a=a)
    result = open_four(game)
    assert result.outcome == "declined" and result.responder is None
    assert "belongs to Agent C" in a.errors[0]
    assert game.getPlayer(A).cashBalance == 1500  # nothing changed
    check_invariants(game)


def test_counter_offer_may_not_switch_partner():
    counter = "give=[Mediterranean Avenue] give_cash=0 get=[] get_cash=150"
    a = ScriptedAgent(A, open=[reply(TO_C)], respond=[reply("[WALK_AWAY]")])
    c = ScriptedAgent(C, respond=[reply(f"[SUBMIT_DEAL] to=[Agent B] {counter}")],  # wrong partner
                      correct=[reply(f"[SUBMIT_DEAL] to=[Agent A] {counter}")])
    game = make_four(a=a, c=c)
    result = open_four(game)
    assert "negotiating with Agent A" in c.errors[0]
    assert result.turns[1]["target"] == A and result.outcome == "walk_away"


def test_rule_based_agent_targets_the_owner():
    game = GameManager(FOUR, agents={n: RuleBasedAgent(n) for n in FOUR})
    give_property(game, A, "Baltic Avenue")
    give_property(game, B, "Reading Railroad")
    give_property(game, C, "Mediterranean Avenue")
    with quiet():
        game.negotiationPhase(game.getPlayer(A))
    result = game.negotiationResults[0]
    assert result.outcome == "deal" and result.responder == C
    assert game.ownsFullSet(A, "Brown")
    check_invariants(game)


def test_llm_open_prompt_lists_partners_and_to():
    sent = []
    real = llm_client.chat
    llm_client.chat = lambda messages, **kw: sent.append([dict(m) for m in messages]) or reply("[PASS]")
    try:
        agent = LLMAgent(A)
        game = make_four(a=agent)
        a = game.getPlayer(A)
        agent.open_negotiation(game, a, game.tradePartners(a))
    finally:
        llm_client.chat = real
    assert "Players you can make an offer to: Agent B, Agent C, Agent D." in sent[0][1]["content"]
    assert "[SUBMIT_DEAL] to=[Agent B]" in sent[0][0]["content"]  # example uses a real name


def test_four_player_mock_games_trade_between_valid_pairs():
    deals = 0
    for seed in range(10):
        random.seed(seed)
        game = GameManager(FOUR, agents={n: RuleBasedAgent(n) for n in FOUR})
        with quiet():
            for _ in range(150):
                game.executeTurn()
                check_invariants(game)
        for r in game.negotiationResults:
            if r.trade:
                deals += 1
                assert r.trade["initiator"] == r.initiator and r.trade["target"] == r.responder
                assert r.initiator != r.responder
    assert deals > 0, "rule bots never traded in 10 four-player games"


def test_four_player_mock_games_keep_ownership_consistent():
    for seed in range(10):
        random.seed(seed)
        game = GameManager(FOUR, agents={n: RuleBasedAgent(n) for n in FOUR})
        with quiet():
            for _ in range(150):
                game.executeTurn()
                check_invariants(game)


def test_four_player_low_cash_games_end_consistently():
    eliminated = 0
    for seed in range(20):
        random.seed(seed)
        game = GameManager(FOUR, agents={n: RuleBasedAgent(n) for n in FOUR})
        for p in game.players:
            p.cashBalance = 150
        with quiet():
            for _ in range(300):
                game.executeTurn()
                check_invariants(game)
                if game.isGameOver():
                    break
            game.finishByTurnLimit()
        survivors = game.activePlayers()
        assert len(game.bankruptcies) == 4 - len(survivors)
        eliminated += len(game.bankruptcies)
        if game.endReason == "bankruptcy":
            assert len(survivors) == 1 and survivors[0].playerName == game.winner
        else:
            assert game.endReason == "turn_limit" and len(survivors) >= 2
            assert game.winner is None or game.winner in [p.playerName for p in survivors]
    assert eliminated >= 5, f"only {eliminated} eliminations in 20 games"


# ------------------------------------------------- call budget (phase 3)
def _offer(n):
    return reply(f"[SUBMIT_DEAL] give=[] give_cash={n} get=[Mediterranean Avenue] get_cash=0")


def _counter(n):
    return reply(f"[SUBMIT_DEAL] give=[Mediterranean Avenue] give_cash=0 get=[] get_cash={n}")


def test_call_cap_ends_negotiation_without_asking():
    a = ScriptedAgent(A, open=[_offer(10)], respond=[_offer(20), _offer(30)])
    b = ScriptedAgent(B, respond=[_counter(500), _counter(400), _counter(300)])
    game = make_game(a, b, max_turns=10)
    game.maxCallsPerTurn = 3
    result = run(game)
    assert result.outcome == "call_cap" and len(result.turns) == 4
    last = result.turns[-1]
    assert last["speaker"] == B and last["capped"] and last["attempts"] == 0
    assert last["kind"] == "REJECT_DEAL"  # the safe default; it is not acted on
    assert len(b.q["respond"]) == 2  # B was asked only once: the capped call never happened
    assert game.callsThisTurn == 3
    assert game.getPlayer(A).cashBalance == 1500 and game.getPlayer(B).cashBalance == 1500
    check_invariants(game)


def test_call_cap_blocks_the_retry():
    fake = "[SUBMIT_DEAL] give=[Reading Railroad] give_cash=0 get=[] get_cash=50"  # A does not own it
    a = ScriptedAgent(A, open=[reply(fake)], correct=[reply(OFFER)])
    game = make_game(a, ScriptedAgent(B))
    game.maxCallsPerTurn = 1
    result = run(game)
    turn = result.turns[0]
    assert result.outcome == "call_cap"
    assert turn["capped"] and turn["fell_back"] and turn["attempts"] == 1 and turn["kind"] == "PASS"
    assert "does not own" in turn["error"]
    assert a.errors == []  # correct() was never called
    check_invariants(game)


def test_capped_buy_decision_passes_without_asking():
    from negotiation import ask_buy_decision

    agent = ScriptedAgent(A, buy=[reply("[BUY]")])
    game = make_game(agent)
    game.maxCallsPerTurn = 2
    game.callsThisTurn = 2
    with quiet():
        bought = ask_buy_decision(game, agent, game.getPlayer(A), game.board.getTile(tile_id("Vermont Avenue")))
    assert bought is False and agent.q["buy"] == [reply("[BUY]")]
    assert game.decisionLog[0]["capped"] and game.decisionLog[0]["attempts"] == 0


def test_call_budget_resets_every_dice_roll():
    agent = ScriptedAgent(A, buy=[reply("[BUY]")])
    game = GameManager([A, B], agents={A: agent}, maxCallsPerTurn=1)
    game.callsThisTurn = 1  # used up "last turn"
    game.rollDice = lambda: (tile_id("Vermont Avenue"), False)
    with quiet():
        game.executeTurn()
    assert tile_id("Vermont Avenue") in game.getPlayer(A).ownedProperties  # the buy got budget
    assert game.callsThisTurn == 1


def test_zero_or_none_means_no_call_limit():
    for limit in (0, None):
        game = GameManager([A, B], maxCallsPerTurn=limit)
        assert all(game.useAgentCall() for _ in range(500))


def test_summary_reports_the_call_cap():
    import main

    a = ScriptedAgent(A, open=[_offer(10)], respond=[_offer(20)])
    b = ScriptedAgent(B, respond=[_counter(500), _counter(400)])
    game = make_game(a, b, max_turns=10)
    game.maxCallsPerTurn = 2
    game.negotiationResults.append(run(game))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main.print_summary(game, stopped_early=False)
    text = out.getvalue()
    assert "Call cap (2 per turn) reached 1 time(s): 1 decision(s) not asked" in text
    assert "Agent replies: 2 " in text  # the capped turn is not counted as a reply
    assert "'call_cap': 1" in text


def test_four_player_mock_games_respect_the_call_cap():
    for cap in (1, 3, 12):
        random.seed(cap)
        game = GameManager(FOUR, agents={n: RuleBasedAgent(n) for n in FOUR}, maxCallsPerTurn=cap)
        with quiet():
            for _ in range(100):
                game.executeTurn()
                assert game.callsThisTurn <= cap
                check_invariants(game)


# ------------------------------------------------ floating an idea (phase 4)
from actions import CHOOSE_ACTIONS, EVERYONE, IDEA_REPLY_ACTIONS  # noqa: E402

IDEA = "[FLOAT_IDEA] want=[Mediterranean Avenue, Reading Railroad] give=[] give_cash=300 get_cash=0"
C_SELLS = "[SUBMIT_DEAL] to=[Agent A] give=[Mediterranean Avenue] give_cash=0 get=[] get_cash=100"
B_SELLS = "[SUBMIT_DEAL] to=[Agent A] give=[Reading Railroad] give_cash=0 get=[] get_cash=250"
MED, READING, BALTIC = tile_id("Mediterranean Avenue"), tile_id("Reading Railroad"), tile_id("Baltic Avenue")


def test_parse_float_idea_and_new_tags():
    def act(text, allowed):
        return parse_action(parse_reply(reply(text)), allowed, BOARD)

    idea = act("[FLOAT_IDEA] want=[Mediterranean Avenue] give=[Baltic Avenue] give_cash=10 get_cash=0",
               OPEN_ACTIONS)
    assert idea.kind == "FLOAT_IDEA"
    assert (idea.deal.get_props, idea.deal.give_props, idea.deal.give_cash) == ((MED,), (BALTIC,), 10)
    assert act("[FLOAT_IDEA] get=[Mediterranean Avenue] give_cash=5", OPEN_ACTIONS).deal.get_props == (MED,)
    assert "idea is empty" in act("[FLOAT_IDEA] want=[] give=[] give_cash=0 get_cash=0", OPEN_ACTIONS).error
    assert "Could not read the idea" in act("[FLOAT_IDEA] please", OPEN_ACTIONS).error
    assert "not allowed" in act("[FLOAT_IDEA] want=[Baltic] give_cash=5", RESPOND_ACTIONS).error
    assert "not allowed" in act("[IGNORE]", OPEN_ACTIONS).error
    assert act("[IGNORE]", IDEA_REPLY_ACTIONS).kind == "IGNORE"
    assert act("[ACCEPT_DEAL] from=[Agent C]", CHOOSE_ACTIONS).source == "Agent C"


def test_validate_idea_rules():
    game = make_four()
    a = game.getPlayer(A)

    def check(give=(), give_cash=0, want=(), get_cash=0):
        return game.validateIdea(a, Deal(tuple(give), give_cash, tuple(want), get_cash))

    assert check(want=[MED], give_cash=90) is None
    assert check(want=[MED, READING], give=[BALTIC]) is None  # wanted from two different owners
    assert "unowned" in check(want=[tile_id("Vermont Avenue")])
    assert "already own" in check(want=[BALTIC])
    assert "do not own" in check(give=[READING], want=[MED])
    assert "only have" in check(give_cash=99999, want=[MED])
    assert "must say what you want" in check(give=[BALTIC])


def test_lock_rules():
    game = make_four()
    game.lockOffer(A, C, [BALTIC])
    assert game.lockConflict(A, C, [BALTIC]) is None  # same offer, same recipient
    assert "promised to Agent C" in game.lockConflict(A, D, [BALTIC])  # promising it twice
    assert "promised to Agent C" in game.lockConflict(A, EVERYONE, [BALTIC])
    game.lockOffer(B, EVERYONE, [READING])  # an idea promises to everyone...
    assert game.lockConflict(B, D, [READING]) is None  # ...so any one recipient is fine
    game.releaseLocks(owner=A)
    assert game.locks == {READING: (B, EVERYONE)}
    game.releaseLocks()
    assert game.locks == {}


def test_broadcast_sealed_offers_accept_one():
    a = ScriptedAgent(A, open=[reply(IDEA, "Who will sell?")], choose=[reply("[ACCEPT_DEAL] from=[Agent C]")])
    b = ScriptedAgent(B, idea=[reply(B_SELLS, "B-PRIVATE-PITCH")])
    c = ScriptedAgent(C, idea=[reply(C_SELLS, "C-PRIVATE-PITCH")])
    d = ScriptedAgent(D, idea=[reply("[IGNORE]")])
    game = make_four(a=a, b=b, c=c, d=d)
    result = open_four(game)

    assert result.kind == "broadcast" and result.outcome == "deal"
    assert result.responder == C and result.proposer == C and set(result.offers) == {B, C}
    # Only the chosen trade happened.
    assert game.getPlayer(A).cashBalance == 1400 and game.getPlayer(C).cashBalance == 1600
    assert MED in game.getPlayer(A).ownedProperties
    assert READING in game.getPlayer(B).ownedProperties and game.getPlayer(B).cashBalance == 1500
    # Sealed: every responder saw the same idea text, and nobody saw another reply.
    assert b.incoming[0] == c.incoming[0] == d.incoming[0]
    assert "Mediterranean Avenue (owned by Agent C)" in d.incoming[0]
    assert "PITCH" not in d.incoming[0] and "PITCH" not in c.incoming[0]
    # The proposer saw both offers from its own point of view, and they were locked.
    assert "C-PRIVATE-PITCH" in a.incoming[0] and "B-PRIVATE-PITCH" in a.incoming[0]
    assert "Agent D did not make an offer" in a.incoming[0]
    assert a.locks_seen == {MED: (C, A), READING: (B, A)}
    assert game.locks == {}  # all released at the end
    check_invariants(game)


def test_idea_give_properties_are_locked_to_everyone():
    idea = "[FLOAT_IDEA] want=[Mediterranean Avenue] give=[Baltic Avenue] give_cash=0 get_cash=0"
    swap = "[SUBMIT_DEAL] to=[Agent A] give=[Mediterranean Avenue] give_cash=0 get=[Baltic Avenue] get_cash=0"
    a = ScriptedAgent(A, open=[reply(idea)], choose=[reply("[WALK_AWAY]")])
    c = ScriptedAgent(C, idea=[reply(swap)])
    game = make_four(a=a, c=c)
    result = open_four(game)
    assert a.locks_seen == {BALTIC: (A, EVERYONE), MED: (C, A)}
    assert result.outcome == "walk_away" and game.locks == {}
    assert BALTIC in game.getPlayer(A).ownedProperties  # walking away changes nothing
    check_invariants(game)


def test_broadcast_counter_continues_as_two_players():
    counter = "[SUBMIT_DEAL] to=[Agent C] give=[] give_cash=120 get=[Mediterranean Avenue] get_cash=0"
    expensive = "[SUBMIT_DEAL] to=[Agent A] give=[Mediterranean Avenue] give_cash=0 get=[] get_cash=150"
    a = ScriptedAgent(A, open=[reply(IDEA)], choose=[reply(counter, "How about $120?")])
    c = ScriptedAgent(C, idea=[reply(expensive)], respond=[reply("[ACCEPT_DEAL]", "Fine.")])
    game = make_four(a=a, c=c)
    result = open_four(game)
    assert result.outcome == "deal" and result.responder == C and result.proposer == A
    assert game.getPlayer(A).cashBalance == 1380 and game.getPlayer(C).cashBalance == 1620
    assert "How about $120?" in c.incoming[1] and "YOU would get: $120" in c.incoming[1]
    assert game.locks == {}
    check_invariants(game)


def test_broadcast_with_no_offers():
    a = ScriptedAgent(A, open=[reply(IDEA)])
    game = make_four(a=a)  # B, C and D use the default: ignore ideas
    result = open_four(game)
    assert result.outcome == "no_offers" and result.offers == {} and game.locks == {}
    assert [t["kind"] for t in result.turns] == ["FLOAT_IDEA", "IGNORE", "IGNORE", "IGNORE"]


def test_invalid_responder_offer_retries_then_ignores():
    bad = "[SUBMIT_DEAL] to=[Agent A] give=[Baltic Avenue] give_cash=0 get=[] get_cash=50"  # A's property
    a = ScriptedAgent(A, open=[reply(IDEA)])
    c = ScriptedAgent(C, idea=[reply(bad)], correct=[reply(bad)])
    game = make_four(a=a, c=c)
    result = open_four(game)
    c_turn = result.turns[2]
    assert c_turn["speaker"] == C and c_turn["kind"] == "IGNORE" and c_turn["fell_back"]
    assert "belongs to Agent A" in c.errors[0]
    assert result.outcome == "no_offers" and game.locks == {}


def test_responder_must_address_the_proposer():
    a = ScriptedAgent(A, open=[reply(IDEA)], choose=[reply("[WALK_AWAY]")])
    c = ScriptedAgent(C, idea=[reply(C_SELLS.replace("to=[Agent A]", "to=[Agent B]"))],
                      correct=[reply(C_SELLS)])
    game = make_four(a=a, c=c)
    result = open_four(game)
    assert "negotiating with Agent A" in c.errors[0]
    assert C in result.offers and result.outcome == "walk_away"


def test_choosing_must_name_a_real_offer():
    a = ScriptedAgent(A, open=[reply(IDEA)], choose=[reply("[ACCEPT_DEAL]")],
                      correct=[reply("[ACCEPT_DEAL] from=[Agent D]")])  # D made no offer
    b = ScriptedAgent(B, idea=[reply(B_SELLS)])
    c = ScriptedAgent(C, idea=[reply(C_SELLS)])
    game = make_four(a=a, b=b, c=c)
    result = open_four(game)
    assert "Say which offer you mean" in a.errors[0]
    choose = result.turns[-1]
    assert choose["fell_back"] and "no offer from 'Agent D'" in choose["error"]
    assert result.outcome == "walk_away" and game.getPlayer(A).cashBalance == 1500  # safe default


def test_single_offer_can_be_accepted_without_from():
    a = ScriptedAgent(A, open=[reply(IDEA)], choose=[reply("[ACCEPT_DEAL]")])
    c = ScriptedAgent(C, idea=[reply(C_SELLS)])
    game = make_four(a=a, c=c)
    assert open_four(game).outcome == "deal" and MED in game.getPlayer(A).ownedProperties


def test_broadcast_respects_the_call_cap():
    # open (1) + three replies (3) = 4 calls: the proposer's choice is capped.
    a = ScriptedAgent(A, open=[reply(IDEA)], choose=[reply("[ACCEPT_DEAL] from=[Agent C]")])
    c = ScriptedAgent(C, idea=[reply(C_SELLS)])
    game = make_four(a=a, c=c)
    game.maxCallsPerTurn = 4
    result = open_four(game)
    assert result.outcome == "call_cap" and result.turns[-1]["capped"]
    assert a.q["choose"]  # never asked
    assert game.getPlayer(A).cashBalance == 1500 and game.locks == {}
    # With only 2 calls, C and D are never asked: no offers because of the cap.
    a2 = ScriptedAgent(A, open=[reply(IDEA)])
    game2 = make_four(a=a2)
    game2.maxCallsPerTurn = 2
    assert open_four(game2).outcome == "call_cap"


def test_rule_based_agent_floats_an_idea_and_deals():
    agents = {n: RuleBasedAgent(n) for n in FOUR}
    agents[A] = RuleBasedAgent(A, float_ideas=True)
    game = GameManager(FOUR, agents=agents)
    give_property(game, A, "Baltic Avenue")
    give_property(game, B, "Reading Railroad")
    give_property(game, C, "Mediterranean Avenue")
    with quiet():
        game.negotiationPhase(game.getPlayer(A))
    result = game.negotiationResults[0]
    assert result.kind == "broadcast" and result.outcome == "deal" and result.responder == C
    assert list(result.offers) == [C]  # only C owns what A wants
    assert game.ownsFullSet(A, "Brown") and game.locks == {}
    check_invariants(game)


def test_llm_agent_idea_prompts():
    sent = []
    real = llm_client.chat
    llm_client.chat = lambda messages, **kw: sent.append([dict(m) for m in messages]) or reply("[IGNORE]")
    try:
        agent = LLMAgent(C)
        game = make_four(c=agent)
        a, c = game.getPlayer(A), game.getPlayer(C)
        agent.respond_to_idea(game, c, a, "Agent A floats an idea to every player: ...", Deal((), 90, (MED,), 0))
        agent.end_conversation()
        agent.choose_offer(game, c, "Replies to your idea:\n- Agent B says ...", {B: Deal()})
    finally:
        llm_client.chat = real
    idea_msg, choose_msg = sent[0][1]["content"], sent[1][1]["content"]
    assert "[SUBMIT_DEAL] to=[Agent A]" in idea_msg and "[IGNORE]" in idea_msg
    assert "Replies to your idea" in choose_msg and "[ACCEPT_DEAL] from=[Player]" in choose_msg
    assert "[FLOAT_IDEA]" in sent[0][0]["content"]  # documented in the system prompt


def test_four_player_mock_games_with_ideas_and_offers():
    kinds = {"targeted": 0, "broadcast": 0}
    for seed in range(10):
        random.seed(seed)
        agents = {n: RuleBasedAgent(n, float_ideas=i % 2 == 1) for i, n in enumerate(FOUR)}
        game = GameManager(FOUR, agents=agents, maxCallsPerTurn=12)
        with quiet():
            for _ in range(150):
                game.executeTurn()
                check_invariants(game)
                assert game.locks == {}  # nothing stays promised between turns
        for r in game.negotiationResults:
            if r.trade:
                kinds[r.kind] += 1
                assert {r.trade["initiator"], r.trade["target"]} == {r.initiator, r.responder}
    assert kinds["targeted"] > 0 and kinds["broadcast"] > 0, kinds


# --------------------------------------------- per-player stats (phase 5)
def _negotiate(game, opener):
    from negotiation import run_negotiation

    with quiet():
        result = run_negotiation(game, game.getPlayer(opener), None, game.maxNegotiationTurns)
    game.negotiationResults.append(result)
    return result


def _three_deal_game():
    """A, B, C, D with three known deals (numbers worked out by hand; SET_BONUS_RATE 0.6):

    1. A buys Mediterranean from C for $90: completes A's Brown set.
       A: strategic +42 (gains the $72 set bonus) but net worth -30.  C: +30 / +30.
    2. D buys Reading Railroad from B for $200: exactly the price, one railroad, no bonus.
       B: 0 / 0 and D: 0 / 0 (zero-gain deals).
    3. C buys Baltic from A for $50: A breaks its own set.  A: -82 / -10.  C: +10 / +10.
    Plus B makes one buy decision (Vermont Avenue).
    """
    a = ScriptedAgent(A, open=[reply(TO_C)], respond=[reply("[ACCEPT_DEAL]")])
    b = ScriptedAgent(B, buy=[reply("[BUY]")], respond=[reply("[ACCEPT_DEAL]")])
    c = ScriptedAgent(C, respond=[reply("[ACCEPT_DEAL]")],
                      open=[reply("[SUBMIT_DEAL] to=[Agent A] give=[] give_cash=50 get=[Baltic Avenue] get_cash=0")])
    d = ScriptedAgent(D, open=[reply("[SUBMIT_DEAL] to=[Agent B] give=[] give_cash=200 "
                                     "get=[Reading Railroad] get_cash=0")])
    game = make_four(a=a, b=b, c=c, d=d)
    land_on(game, B, tile_id("Vermont Avenue"))
    for opener in (A, D, C):
        assert _negotiate(game, opener).outcome == "deal"
    return game


def test_per_player_stats_numbers():
    import main

    stats = main.per_player_stats(_three_deal_game())
    expected = {  # deals, strategic, net worth, worse strategic, worse net worth, zero gain, opened, calls
        A: (2, -40, -40, 1, 2, 0, 1, 2),
        B: (1, 0, 0, 0, 0, 1, 0, 2),  # 2 calls: the buy decision and accepting D's offer
        C: (2, 40, 40, 0, 0, 0, 1, 2),
        D: (1, 0, 0, 0, 0, 1, 1, 1),
    }
    keys = ("deals", "trade_strategic_change", "trade_net_worth_change", "deals_worse_strategic",
            "deals_worse_net_worth", "deals_zero_strategic_gain", "negotiations_opened", "agent_calls")
    for name, values in expected.items():
        assert tuple(stats[name][k] for k in keys) == values, (name, {k: stats[name][k] for k in keys})
        assert stats[name]["ideas_floated"] == 0 and stats[name]["capped"] == 0
        assert not stats[name]["bankrupt"]


def test_per_player_stats_count_ideas_caps_and_participants():
    import main

    a = ScriptedAgent(A, open=[reply(IDEA)], choose=[reply("[ACCEPT_DEAL] from=[Agent C]")])
    c = ScriptedAgent(C, idea=[reply(C_SELLS)])
    game = make_four(a=a, c=c)
    game.maxCallsPerTurn = 4  # the proposer's choice is capped
    result = _negotiate(game, A)
    assert result.participants == [A, B, C, D]
    stats = main.per_player_stats(game)
    assert stats[A]["ideas_floated"] == 1 and stats[A]["negotiations_opened"] == 1
    assert stats[A]["capped"] == 1 and stats[A]["agent_calls"] == 1  # floated, never asked to choose
    assert stats[B]["agent_calls"] == stats[C]["agent_calls"] == stats[D]["agent_calls"] == 1
    assert all(s["deals"] == 0 for s in stats.values())


def test_targeted_negotiation_participants():
    game = make_four(a=ScriptedAgent(A, open=[reply(TO_C)]), c=ScriptedAgent(C, respond=[reply("[WALK_AWAY]")]))
    assert _negotiate(game, A).participants == [A, C]


def test_summary_and_log_show_per_player_numbers():
    import argparse
    import main

    game = _three_deal_game()
    with quiet():
        game.finishByTurnLimit()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main.print_summary(game, stopped_early=False)
    lines = out.getvalue().splitlines()
    assert "Deals executed: 3" in lines
    row_a = next(line for line in lines if line.strip().startswith(A) and "-40" in line)
    assert row_a.split()[2:6] == ["2", "-40", "-40", "1/2"]
    assert sum(1 for line in lines if line.startswith("  Agent ")) == 4  # one table row per player

    args = argparse.Namespace(mode="mock", players=4, turns=0, seed=None)
    log = main.build_log(game, args, stopped_early=False)
    assert log["per_player"] == main.per_player_stats(game)
    assert log["negotiations"][0]["participants"] == [A, C]
    assert log["negotiations"][0]["kind"] == "targeted"
    json_text = __import__("json").dumps(log)  # the whole log must be JSON-serializable
    assert '"per_player"' in json_text


# --------------------------------------------- live-run fixes: trade events
def test_trade_event_says_what_changed_hands():
    a = ScriptedAgent(A, open=[reply(TO_C)])
    c = ScriptedAgent(C, respond=[reply("[ACCEPT_DEAL]")])
    game = make_four(a=a, c=c)
    open_four(game)
    assert game.events[-1] == "Turn 0: Trade: Agent A gave $90 to Agent C and received Mediterranean Avenue."
    assert not any("completed a trade" in e for e in game.events)


def test_trade_event_for_a_swap_with_cash():
    game = make_game()
    a, b = game.getPlayer(A), game.getPlayer(B)
    swap = {"giveCash": 20, "giveProps": [BALTIC], "receiveCash": 0, "receiveProps": [MED]}
    with quiet():
        game.executeTrade(a, b, swap)
    assert game.events[-1].endswith("Agent A gave Baltic Avenue + $20 to Agent B and received Mediterranean Avenue.")


def test_state_lists_your_own_properties():
    from agents import describe_state

    game = make_four()
    assert "Properties you own: Baltic Avenue." in describe_state(game, game.getPlayer(A))
    assert "Properties you own: none." in describe_state(game, game.getPlayer(D))


# ------------------------------------ live-run fixes: empty / cut-off replies
def test_chat_counts_tokens_empty_and_cut_off_replies():
    def go():
        text = llm_client.chat([{"role": "user", "content": "hi"}])
        return (text, llm_client.last_finish_reason, llm_client.empty_replies, llm_client.cut_off_replies,
                llm_client.total_prompt_tokens, llm_client.total_completion_tokens)

    llm_client.empty_replies = llm_client.cut_off_replies = 0
    llm_client.total_prompt_tokens = llm_client.total_completion_tokens = 0
    (text, finish, empty, cut, sent, produced), _ = _run_with_completions(
        _SeqCompletions([("", "length")], prompt_tokens=1200, completion_tokens=800), go)
    assert (text, finish, empty, cut, sent, produced) == ("", "length", 1, 1, 1200, 800)


def test_chat_remembers_options_the_model_rejected():
    def go():
        llm_client.chat([{"role": "user", "content": "hi"}])
        return llm_client.chat([{"role": "user", "content": "again"}])

    errors = [_bad_request("Unsupported parameter: 'max_tokens'."), _bad_request("'temperature' unsupported")]
    _, calls = _with_fake_client(errors, go)
    assert len(calls) == 4  # 2 rejected + 2 successful; the second chat() needed no retry
    assert "max_completion_tokens" in calls[3] and "max_tokens" not in calls[3]
    assert "temperature" not in calls[3]


def test_empty_reply_gets_an_honest_error():
    a = ScriptedAgent(A, open=[""], correct=[reply(OFFER)])
    b = ScriptedAgent(B, respond=[reply("[ACCEPT_DEAL]")])
    result = run(make_game(a, b))
    assert a.errors[0].startswith("Your reply was empty.")
    assert result.outcome == "deal" and result.turns[0]["attempts"] == 2
    assert result.turns[0]["finish_reasons"] == [None, None]  # scripted agents use no model


def test_cut_off_reply_is_explained_and_retried_with_more_room():
    agent = LLMAgent(A, max_tokens=800)
    game = make_game(agent, ScriptedAgent(B, respond=[reply("[ACCEPT_DEAL]")]))
    completions = _SeqCompletions([("", "length"), (reply(OFFER), "stop")])
    result, completions = _run_with_completions(completions, lambda: run(game))
    turn = result.turns[0]
    assert turn["finish_reasons"] == ["length", "stop"]
    assert "cut off" in turn["retry_errors"][0]
    assert completions.calls[0]["max_tokens"] == 800 and completions.calls[1]["max_tokens"] == 1600
    assert (turn["prompt_tokens"], turn["completion_tokens"]) == (200, 100)
    assert result.outcome == "deal"


def test_per_player_stats_count_cut_off_replies():
    import main

    agent = LLMAgent(A, max_tokens=800)
    game = make_game(agent, ScriptedAgent(B, respond=[reply("[ACCEPT_DEAL]")]))
    completions = _SeqCompletions([("", "length"), (reply(OFFER), "stop")])
    result, _ = _run_with_completions(completions, lambda: run(game))
    game.negotiationResults.append(result)
    stats = main.per_player_stats(game)
    assert stats[A]["cut_off_replies"] == 1 and stats[A]["prompt_tokens"] == 200
    assert stats[B]["cut_off_replies"] == 0 and stats[B]["prompt_tokens"] == 0


# ------------------------------------------------ swap scenario (restored)
def _swap_game(players=(A, B), agents=None):
    from scenarios import apply_scenario

    game = GameManager(list(players), agents=agents or {})
    with quiet():
        apply_scenario(game, "swap")
    return game


def test_swap_scenario_starting_position():
    game = _swap_game(FOUR)
    a, b, c, d = (game.getPlayer(n) for n in FOUR)
    assert sorted(game.board.getTile(i).name for i in a.ownedProperties) == \
        ["Baltic Avenue", "Oriental Avenue", "Vermont Avenue"]
    assert sorted(game.board.getTile(i).name for i in b.ownedProperties) == \
        ["Connecticut Avenue", "Mediterranean Avenue"]
    assert (a.cashBalance, b.cashBalance) == (1240, 1320)
    assert all(game.calculateEquity(p) == 1500 for p in (a, b, c, d))  # same net worth for all
    assert c.ownedProperties == [] and d.ownedProperties == []
    assert "Scenario 'swap'" in game.events[0]
    check_invariants(game)


def test_swap_scenario_values_match_the_documented_numbers():
    game = _swap_game()
    a, b = game.getPlayer(A), game.getPlayer(B)
    swap = Deal((BALTIC,), 0, (tile_id("Connecticut Avenue"),), 0)
    assert game.dealValueDelta(a, swap) == 252 and game.dealValueDelta(b, swap.flipped()) == 12
    with quiet():
        trade = game.executeTrade(a, b, swap.to_offer_data())
    assert (trade["initiator_strategic_delta"], trade["target_strategic_delta"]) == (252, 12)
    assert (trade["initiator_equity_delta"], trade["target_equity_delta"]) == (60, -60)
    assert game.ownsFullSet(A, "Light Blue") and game.ownsFullSet(B, "Brown")
    check_invariants(game)


def test_swap_scenario_needs_a_fresh_game_and_two_players():
    from scenarios import apply_scenario

    game = _swap_game()
    try:
        apply_scenario(game, "swap")
        raise AssertionError("expected an error")
    except ValueError as error:
        assert "already owned" in str(error)
    try:
        apply_scenario(GameManager([A]), "swap")
        raise AssertionError("expected an error")
    except ValueError as error:
        assert "at least 2 players" in str(error)
    apply_scenario(GameManager([A, B]), "none")  # "none" does nothing


def test_swap_scenario_mock_games_trade_and_stay_consistent():
    deals = 0
    for seed in range(5):
        random.seed(seed)
        game = _swap_game(agents={A: RuleBasedAgent(A), B: RuleBasedAgent(B)})
        with quiet():
            for _ in range(40):
                game.executeTurn()
                check_invariants(game)
        deals += sum(1 for r in game.negotiationResults if r.trade)
    assert deals > 0


# ------------------------------------------------------------------ runner
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"  ok    {name}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
