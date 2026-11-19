"""Run a Monopoly game between agents (4 players by default).

Examples (run from the project's top folder):

    python monopoly\\main.py                       # offline: rule-based bots, no API, no tokens
    python monopoly\\main.py --players 2           # the original two-player game
    python monopoly\\main.py --mode llm-vs-rule    # Agent A is an LLM, the others are rule-based bots
    python monopoly\\main.py --mode llm            # every agent is an LLM
    python monopoly\\main.py --mode llm --turns 6 --show-thoughts

Note: with more than 2 players there is no trading yet (it comes with choosing a partner).
"""
import argparse
import json
import os
import random
import time

import llm_client
from agents import PERSONAS, LLMAgent, RuleBasedAgent
from gameManager import GameManager

MAX_PLAYERS = 26  # Agent A ... Agent Z


def player_names(count):
    """["Agent A", "Agent B", ...] for `count` players."""
    return [f"Agent {chr(ord('A') + i)}" for i in range(count)]


def build_agents(args, names):
    """One agent per player name.

    mock:        every player is a rule-based bot that proposes trades.
    llm-vs-rule: Agent A is an LLM; everyone else is a rule-based bot that only responds.
    llm:         every player is an LLM. --persona-a / --persona-b set Agent A / Agent B;
                 other players get no persona.
    """
    personas = {"Agent A": args.persona_a, "Agent B": args.persona_b}
    agents = {}
    for name in names:
        if args.mode == "mock":
            agents[name] = RuleBasedAgent(name, propose=True)
        elif args.mode == "llm-vs-rule" and name != "Agent A":
            agents[name] = RuleBasedAgent(name, propose=False)
        else:
            agents[name] = LLMAgent(name, persona=personas.get(name))
    return agents


def describe_result(game):
    """One line saying who won and why."""
    reason = game.endReason
    if reason == "bankruptcy":
        out = ", ".join(f"{b['player']} went bankrupt on turn {b['turn']}" for b in game.bankruptcies)
        return f"Winner: {game.winner} ({out})"
    if reason == "turn_limit":
        if game.winner:
            return f"Winner: {game.winner} (highest net worth after {game.turnCount} turns)"
        return f"Result: draw (equal net worth after {game.turnCount} turns)"
    if reason == "stopped_early":
        return "Result: no winner (game stopped early)"
    return "Result: game not finished"


def print_summary(game, stopped_early):
    print("\n" + "=" * 60)
    print("GAME SUMMARY" + ("  (stopped early)" if stopped_early else ""))
    print("=" * 60)
    print(f"Turns played: {game.turnCount}")
    print(describe_result(game))
    for player in game.players:
        names = [game.board.getTile(i).name for i in player.ownedProperties] or ["none"]
        print(
            f"{player.playerName}{' (BANKRUPT)' if player.isBankrupt else ''}: cash ${player.cashBalance}, "
            f"net worth ${game.calculateEquity(player)}, "
            f"strategic value ${game.calculateStrategicValue(player)}"
        )
        print(f"    properties: {', '.join(names)}")

    outcomes = {}
    for result in game.negotiationResults:
        outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
    print(f"Negotiation phases: {len(game.negotiationResults)}  {outcomes}")

    all_turns = [t for r in game.negotiationResults for t in r.turns] + game.decisionLog
    turns = [t for t in all_turns if t["attempts"] > 0]  # the agent was actually asked
    if turns:
        bad_format = sum(1 for t in turns if not t["format_ok"])
        fallbacks = sum(1 for t in turns if t["fell_back"])
        retried = sum(1 for t in turns if t["attempts"] > 1)
        print(f"Agent replies: {len(turns)}  (format problems: {bad_format}, "
              f"retried after an invalid reply: {retried}, used default after retry: {fallbacks})")
    capped = [t for t in all_turns if t.get("capped")]
    if capped:
        never_asked = sum(1 for t in capped if t["attempts"] == 0)
        print(f"Call cap ({game.maxCallsPerTurn} per turn) reached {len(capped)} time(s): "
              f"{never_asked} decision(s) not asked, {len(capped) - never_asked} retry(s) skipped")

    # Every executed deal, scored against not trading at all (the BATNA: change of 0).
    deals = [r.trade for r in game.negotiationResults if r.trade]
    if deals:
        print(f"Deals executed: {len(deals)}")
        for player in game.players:
            deltas = []
            for trade in deals:
                if trade["initiator"] == player.playerName:
                    deltas.append(trade["initiator_strategic_delta"])
                elif trade["target"] == player.playerName:
                    deltas.append(trade["target_strategic_delta"])
            worse = sum(1 for d in deltas if d < 0)
            print(f"    {player.playerName}: total strategic change {sum(deltas):+d}; "
                  f"deals that left it worse off than not trading: {worse} of {len(deltas)}")
    if llm_client.total_calls:
        print(f"LLM calls: {llm_client.total_calls}   tokens used: {llm_client.total_tokens_used}")


def save_log(game, args, stopped_early):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.abspath(os.path.join(folder, time.strftime("run_%Y%m%d_%H%M%S.json")))
    data = {
        "settings": {**vars(args), "base_url": llm_client.BASE_URL, "model": llm_client.MODEL},
        "stopped_early": stopped_early,
        "turns_played": game.turnCount,
        **game.gameResult(),  # winner, end_reason, bankruptcies
        "tokens_used": llm_client.total_tokens_used,
        "llm_calls": llm_client.total_calls,
        "buy_decisions": game.decisionLog,
        "negotiations": [r.to_dict() for r in game.negotiationResults],
        "events": game.events,
        "final": {
            p.playerName: {
                "cash": p.cashBalance,
                "bankrupt": p.isBankrupt,
                "properties": [game.board.getTile(i).name for i in p.ownedProperties],
                "net_worth": game.calculateEquity(p),
                "strategic_value": game.calculateStrategicValue(p),
            }
            for p in game.players
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Monopoly negotiation playground")
    parser.add_argument("--mode", choices=["mock", "llm-vs-rule", "llm"], default="mock")
    parser.add_argument("--players", type=int, default=4,
                        help=f"number of players, 2 to {MAX_PLAYERS} (default 4)")
    parser.add_argument("--turns", type=int, default=10, help="number of dice-roll turns to play")
    parser.add_argument("--max-negotiation-turns", type=int, default=6,
                        help="most replies allowed in one negotiation (each reply is one model call)")
    parser.add_argument("--max-calls-per-turn", type=int, default=12,
                        help="most agent calls per dice roll, retries included (0 = no limit). "
                             "When it runs out the agent is not asked and the safe default is used")
    parser.add_argument("--seed", type=int, default=None, help="fix the dice for repeatable games")
    parser.add_argument("--show-thoughts", action="store_true", help="print each agent's private Thought")
    parser.add_argument("--persona-a", choices=list(PERSONAS), default=None)
    parser.add_argument("--persona-b", choices=list(PERSONAS), default=None)
    args = parser.parse_args()
    if not 2 <= args.players <= MAX_PLAYERS:
        parser.error(f"--players must be between 2 and {MAX_PLAYERS}")
    if args.max_calls_per_turn < 0:
        parser.error("--max-calls-per-turn must be 0 (no limit) or more")

    if args.mode != "mock" and not os.getenv(llm_client.API_KEY_ENV):
        print(f"{llm_client.API_KEY_ENV} is not set. Add it to your .env file, or run with --mode mock.")
        return

    if args.seed is not None:
        random.seed(args.seed)

    print(f"--- Booting Monopoly Playground Engine (mode: {args.mode}) ---")
    names = player_names(args.players)
    game = GameManager(
        names,
        agents=build_agents(args, names),
        maxNegotiationTurns=args.max_negotiation_turns,
        showThoughts=args.show_thoughts,
        maxCallsPerTurn=args.max_calls_per_turn,
    )

    stopped_early = False
    try:
        for _ in range(args.turns):
            game.executeTurn()
            if game.isGameOver():
                break  # someone went bankrupt
        game.finishByTurnLimit()  # does nothing if the game already ended
    except llm_client.LLMError as error:
        stopped_early = True
        game.markStoppedEarly()
        print(f"\n[STOPPED] The model could not be used: {error}")

    print_summary(game, stopped_early)
    print(f"Log saved to: {save_log(game, args, stopped_early)}")


if __name__ == "__main__":
    main()
