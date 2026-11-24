"""Starting positions for experiments (main.py --scenario NAME).

A scenario hands out properties before turn 1. Each player pays the normal price from
their starting cash, so everyone still starts with the same net worth ($1500); only the
mix of cash and property differs. Players are taken in turn order (the first player is
"Agent A"), so a scenario works with any player names.

swap (needs 2+ players; any others start with nothing):
    Agent A: Oriental Avenue, Vermont Avenue, Baltic Avenue   (2 of 3 Light Blue, 1 of 2 Brown)
    Agent B: Connecticut Avenue, Mediterranean Avenue          (1 of 3 Light Blue, 1 of 2 Brown)
    Swapping Baltic for Connecticut completes a set for BOTH players. With the placeholder
    reward (SET_BONUS_RATE = 0.6) the swap alone is worth +252 strategic value to A and +12
    to B, but by net worth it is +60 for A and -60 for B (Connecticut costs more than Baltic).
    So there is a real surplus and room to bargain over a cash premium paid by A: the deal
    helps both by strategic value for any premium from $0 to $251, and B also stops losing
    net worth once the premium reaches $60.
"""
from tile import PropertyTile

SWAP = {
    0: ["Oriental Avenue", "Vermont Avenue", "Baltic Avenue"],
    1: ["Connecticut Avenue", "Mediterranean Avenue"],
}

SCENARIOS = {
    "swap": (SWAP, "A and B each complete a set by swapping Baltic for Connecticut"),
}


def _tile_by_name(game, name):
    return next(t for t in game.board.tiles if isinstance(t, PropertyTile) and t.name == name)


def apply_scenario(game, name):
    """Set up the named scenario on a fresh game. "none" (or None) does nothing."""
    if name in (None, "none"):
        return
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Choose from: {', '.join(SCENARIOS)}.")
    holdings, _ = SCENARIOS[name]
    if len(game.players) <= max(holdings):
        raise ValueError(f"Scenario '{name}' needs at least {max(holdings) + 1} players.")

    parts = []
    for index, prop_names in holdings.items():
        player = game.players[index]
        for prop_name in prop_names:
            tile = _tile_by_name(game, prop_name)
            if tile.ownerId is not None:
                raise ValueError(f"{prop_name} is already owned; apply a scenario to a fresh game.")
            if not player.deductFunds(tile.price):
                raise ValueError(f"{player.playerName} cannot afford {prop_name}.")
            tile.ownerId = player.playerName
            player.addProperty(tile.index)
        parts.append(f"{player.playerName} starts with {', '.join(prop_names)}")
    game.logEvent(f"Scenario '{name}': {'; '.join(parts)} (paid for from starting cash).")
    print(f"  -> SCENARIO {name}: {'; '.join(parts)}.")
