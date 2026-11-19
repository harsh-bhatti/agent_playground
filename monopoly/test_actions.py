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

    def __init__(self, name, buy=(), open=(), respond=(), correct=()):
        super().__init__(name)
        self.q = {"buy": list(buy), "open": list(open), "respond": list(respond), "correct": list(correct)}
        self.errors = []  # problems the game reported back to us
        self.incoming = []  # messages the game forwarded to us

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


def _with_fake_client(errors, func):
    import types

    completions = _FakeCompletions(errors)
    real_client = llm_client._client
    llm_client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    try:
        with quiet():
            result = func()
    finally:
        llm_client._client = real_client
    return result, completions.calls


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
