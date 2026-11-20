"""Run a Monopoly game between two agents.

Examples (run from the project's top folder):

    python monopoly\\main.py                       # offline: two rule-based bots, no API, no tokens
    python monopoly\\main.py --mode llm-vs-rule    # Agent A is an LLM, Agent B is a rule-based bot
    python monopoly\\main.py --mode llm            # both agents are LLMs
    python monopoly\\main.py --mode llm --turns 6 --show-thoughts
"""
import argparse
import json
import os
import random
import time

import llm_client
from agents import PERSONAS, LLMAgent, RuleBasedAgent
from gameManager import GameManager

NAMES = ["Agent A", "Agent B"]


def build_agents(args):
    if args.mode == "mock":
        return {
            "Agent A": RuleBasedAgent("Agent A", propose=True),
            "Agent B": RuleBasedAgent("Agent B", propose=True),
        }
    if args.mode == "llm-vs-rule":
        return {
            "Agent A": LLMAgent("Agent A", persona=args.persona_a),
            "Agent B": RuleBasedAgent("Agent B", propose=False),
        }
    return {
        "Agent A": LLMAgent("Agent A", persona=args.persona_a),
        "Agent B": LLMAgent("Agent B", persona=args.persona_b),
    }


def print_summary(game, stopped_early):
    print("\n" + "=" * 60)
    print("GAME SUMMARY" + ("  (stopped early)" if stopped_early else ""))
    print("=" * 60)
    print(f"Turns played: {game.turnCount}")
    for player in game.players:
        names = [game.board.getTile(i).name for i in player.ownedProperties] or ["none"]
        print(
            f"{player.playerName}: cash ${player.cashBalance}, "
            f"net worth ${game.calculateEquity(player)}, "
            f"strategic value ${game.calculateStrategicValue(player)}"
        )
        print(f"    properties: {', '.join(names)}")

    outcomes = {}
    for result in game.negotiationResults:
        outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
    print(f"Negotiation phases: {len(game.negotiationResults)}  {outcomes}")

    turns = [t for r in game.negotiationResults for t in r.turns] + game.decisionLog
    if turns:
        bad_format = sum(1 for t in turns if not t["format_ok"])
        fallbacks = sum(1 for t in turns if t["fell_back"])
        retried = sum(1 for t in turns if t["attempts"] > 1)
        print(f"Agent replies: {len(turns)}  (format problems: {bad_format}, "
              f"retried after an invalid reply: {retried}, used default after retry: {fallbacks})")

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
        "settings": {**vars(args), "model": llm_client.MODEL},
        "stopped_early": stopped_early,
        "turns_played": game.turnCount,
        "tokens_used": llm_client.total_tokens_used,
        "llm_calls": llm_client.total_calls,
        "buy_decisions": game.decisionLog,
        "negotiations": [r.to_dict() for r in game.negotiationResults],
        "events": game.events,
        "final": {
            p.playerName: {
                "cash": p.cashBalance,
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
    parser.add_argument("--turns", type=int, default=10, help="number of dice-roll turns to play")
    parser.add_argument("--max-negotiation-turns", type=int, default=6,
                        help="most replies allowed in one negotiation (each reply is one model call)")
    parser.add_argument("--seed", type=int, default=None, help="fix the dice for repeatable games")
    parser.add_argument("--show-thoughts", action="store_true", help="print each agent's private Thought")
    parser.add_argument("--persona-a", choices=list(PERSONAS), default=None)
    parser.add_argument("--persona-b", choices=list(PERSONAS), default=None)
    args = parser.parse_args()

    if args.mode != "mock" and not os.getenv(llm_client.API_KEY_ENV):
        print(f"{llm_client.API_KEY_ENV} is not set. Add it to your .env file, or run with --mode mock.")
        return

    if args.seed is not None:
        random.seed(args.seed)

    print(f"--- Booting Monopoly Playground Engine (mode: {args.mode}) ---")
    game = GameManager(
        NAMES,
        agents=build_agents(args),
        maxNegotiationTurns=args.max_negotiation_turns,
        showThoughts=args.show_thoughts,
    )

    stopped_early = False
    try:
        for _ in range(args.turns):
            game.executeTurn()
    except llm_client.LLMError as error:
        stopped_early = True
        print(f"\n[STOPPED] The model could not be used: {error}")

    print_summary(game, stopped_early)
    print(f"Log saved to: {save_log(game, args, stopped_early)}")


if __name__ == "__main__":
    main()
