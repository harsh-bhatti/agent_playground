# Team update: 2026-11-19

What changed in the Monopoly negotiation engine today, what the live runs showed, and what
still needs a team decision. Everything is on branch `run-logs` (code + the run logs); the code
alone is on `live-run-fixes`. 105 offline tests pass (`python monopoly\test_actions.py`).

## Rule changes you will notice
1. **Bankruptcy.** A player who cannot pay rent or tax in full is out. Owed to a player: that
   player gets all their cash and properties. Owed to the bank: properties go back unowned.
   Tax is now really charged (before, a short player silently paid $0).
2. **Game end.** Last player standing wins. At the turn limit the highest **score** wins.
3. **Score = cash + property prices + set bonus** for every color group you own completely
   (Brown +$72, Railroad +$240, Light Blue +$192; 60% of the group's price, still a
   placeholder: `SET_BONUS_RATE` in `gameManager.py`). The score is the agents' stated goal,
   the research measure and the winner's measure.
4. **4 players by default** (`--players 2` gives the old game).
5. **Negotiation:** only the active player may open one negotiation per dice roll, either
   - a targeted offer: `[SUBMIT_DEAL] to=[Agent C] give=[...] give_cash=N get=[...] get_cash=M`, or
   - an idea floated to everyone: `[FLOAT_IDEA] want=[...] give=[...] give_cash=N get_cash=M`.
     Each other player replies with an offer or `[IGNORE]` without seeing the other replies;
     the proposer then accepts one (`[ACCEPT_DEAL] from=[Agent X]`), counters one, or walks away.
   Properties in an open offer are locked so they cannot be promised to two players.
6. **Call budget:** at most 12 agent replies per dice roll (`--max-calls-per-turn`), retries
   included. When it runs out, the safe default is used (outcome `call_cap`).

The design rule is unchanged: agents only propose; the engine validates; only validated actions
change the game.

## If you wrote your own agent
`Agent.open_negotiation` now receives a list of possible partners. Two new methods,
`respond_to_idea` and `choose_offer`, have safe defaults (ignore / walk away).

## Other changes
- Model: ASU CreateAI (`openai/gpt6_luna`) through `llm_client.py`; Groq was removed.
  Put `CREATEAI_API_KEY` in your own `.env` (see `.env.example`).
- Trade events now say what changed hands; each agent sees "Properties you own".
- Empty model replies fixed (CreateAI reports a cut-off reply as normal; the reply limit is now
  1600 tokens and cut-offs are detected by tokens used).
- `--scenario swap`: A and B start so that swapping Baltic for Connecticut completes a set for
  both (+252 / +12 score), a controlled test of who captures the surplus.
- Summary and log are per player (deals, score and net-worth change, deals worse off than not
  trading, calls, tokens).

## What the live runs showed (logs in `logs/`)
| Log | Setup | Result |
|---|---|---|
| `archive_2player_groq/` (8 runs, Oct) | 2 players, Groq | Models overpaid, accepted zero-gain deals, churned one railroad back and forth. |
| `run_20261119_150437` | 4 LLMs, 8 turns | Partner choice and a floated idea worked; first real haggling; buyers overpaid. |
| `run_20261119_151724`, `run_20261119_152227` | swap scenario, 2 LLMs | Both found the swap at once and split the surplus (A paid $100 / $60). Then B sold its completed set for a small cash gain. |
| `run_20261119_152501` | 4 LLMs, 8 turns | No invalid actions, no empty replies, but no deals: nobody would pay over price. |

Main finding: agents optimize exactly the goal the prompt gives them. Told "net worth", they
sold completed sets; that is why the goal is now the score. **All runs above used the old
net-worth goal**, so do not pool them with runs after this change. Behavior also varies a lot
between runs; one seed proves little.

Cost: about 1,500-2,100 tokens per call (mostly prompt), 45-75k tokens per 8-turn game.

## Decisions still open
- Should agents be shown each offer's computed value? (Today they see the score rule and all
  scores, but not the value of a specific offer.)
- The real reward (replace the 0.6 placeholder).
- Token budget, and how many seeds per experiment (single runs vary too much).

## Try it
```
.\.venv\Scripts\Activate.ps1
python monopoly\test_actions.py                                   # offline, free
python monopoly\main.py --players 4 --turns 60 --seed 1           # offline rule-based bots
python monopoly\main.py --players 2 --mode llm --scenario swap --turns 8 --seed 1   # costs tokens
```
