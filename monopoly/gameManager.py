import random
import json
from player import Player
from board import Board
from tile import PropertyTile
from negotiation import run_negotiation, ask_buy_decision

# PLACEHOLDER for the research reward: how much extra "strategic value" a complete color
# set is worth, as a fraction of the set's total price. calculateEquity() (cash + prices)
# stays as the team's plain net-worth number; this one also credits set completion, which is
# what makes trading for a set worthwhile. We will replace it with the real reward later.
SET_BONUS_RATE = 0.6


class GameManager:
    def __init__(self, playerNames, agents=None, maxNegotiationTurns=6, showThoughts=False):
        self.players = [Player(name) for name in playerNames]
        self.board = Board()
        self.activePlayerIndex = 0

        # agents: {"Agent A": <agent>, ...}. A player with no agent plays the old automatic way.
        self.agents = agents or {}
        self.maxNegotiationTurns = maxNegotiationTurns
        self.showThoughts = showThoughts

        self.turnCount = 0
        self.events = []  # short plain-English history, shown to agents
        self.decisionLog = []  # every agent decision, saved for research
        self.negotiationResults = []

    # ------------------------------------------------------------------ basics
    def rollDice(self):
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        return die1 + die2, die1 == die2

    def getPlayer(self, name):
        return next(p for p in self.players if p.playerName == name)

    def logEvent(self, text):
        self.events.append(f"Turn {self.turnCount}: {text}")

    def executeTurn(self):
        self.turnCount += 1
        activePlayer = self.players[self.activePlayerIndex]
        rollTotal, isDouble = self.rollDice()

        print(f"\n{activePlayer.playerName} rolled a {rollTotal} {'(Doubles!)' if isDouble else ''}")
        self.resolveMovement(activePlayer, rollTotal)
        self.negotiationPhase(activePlayer)

        if not isDouble:
            self.activePlayerIndex = (self.activePlayerIndex + 1) % len(self.players)

    def resolveMovement(self, player, steps):
        player.moveSpaces(steps)
        landedTile = self.board.getTile(player.boardPosition)
        print(f"  -> Landed on: {landedTile.name} (Space {landedTile.index})")
        landedTile.onLand(player, self)

    # ------------------------------------------------------------ buy decision
    def promptBuyDecision(self, player, propertyTile):
        if player.cashBalance < propertyTile.price:
            print(f"  -> ACTION: {player.playerName} cannot afford {propertyTile.name}.")
            return

        agent = self.agents.get(player.playerName)
        wantsToBuy = True  # players without an agent always buy (the original behavior)
        if agent is not None:
            wantsToBuy = ask_buy_decision(self, agent, player, propertyTile)

        if wantsToBuy and player.deductFunds(propertyTile.price):
            propertyTile.ownerId = player.playerName
            player.addProperty(propertyTile.index)
            print(f"  -> ACTION: {player.playerName} bought {propertyTile.name} for ${propertyTile.price}.")
            self.logEvent(f"{player.playerName} bought {propertyTile.name} for ${propertyTile.price}.")
        else:
            print(f"  -> ACTION: {player.playerName} passed on {propertyTile.name}.")
            self.logEvent(f"{player.playerName} passed on buying {propertyTile.name}.")

    def processRentPayment(self, payingPlayer, receivingPlayerId, amount):
        receivingPlayer = self.getPlayer(receivingPlayerId)
        if payingPlayer.deductFunds(amount):
            receivingPlayer.addFunds(amount)
            print(f"  -> TRANSACTION: {payingPlayer.playerName} paid ${amount} rent to {receivingPlayerId}.")
            self.logEvent(f"{payingPlayer.playerName} paid ${amount} rent to {receivingPlayerId}.")
        else:
            print(f"  -> BANKRUPTCY: {payingPlayer.playerName} cannot afford ${amount} rent!")
            self.logEvent(f"{payingPlayer.playerName} could not afford ${amount} rent.")

    # ------------------------------------------------------------ rent & sets
    def colorGroups(self):
        """{"Brown": [tile, tile], ...} for every property on the board."""
        groups = {}
        for tile in self.board.tiles:
            if isinstance(tile, PropertyTile):
                groups.setdefault(tile.colorGroup, []).append(tile)
        return groups

    def ownsFullSet(self, playerName, colorGroup):
        tiles = self.colorGroups()[colorGroup]
        return all(t.ownerId == playerName for t in tiles)

    def getRent(self, tile):
        """Current rent for a property.

        - Railroads: rent grows with how many railroads the owner has.
        - Everything else: rent doubles (with no houses) if the owner holds the whole color set.
        """
        if tile.ownerId is None:
            return tile.rentLevels[0]
        group = self.colorGroups()[tile.colorGroup]
        if tile.colorGroup == "Railroad":
            owned = sum(1 for t in group if t.ownerId == tile.ownerId)
            return tile.rentLevels[min(owned, len(tile.rentLevels)) - 1]
        rent = tile.rentLevels[tile.houseCount]
        if tile.houseCount == 0 and all(t.ownerId == tile.ownerId for t in group):
            rent *= 2
        return rent

    # ------------------------------------------------- reward & equity math
    def calculateEquity(self, player):
        """Calculates total net worth: Cash + (Property Values)"""
        equity = player.cashBalance
        for propId in player.ownedProperties:
            equity += self.board.getTile(propId).price
        return equity

    def _strategicValueFor(self, cash, ownedIds):
        owned = set(ownedIds)
        value = cash + sum(self.board.getTile(i).price for i in owned)
        for tiles in self.colorGroups().values():
            if all(t.index in owned for t in tiles):
                value += int(SET_BONUS_RATE * sum(t.price for t in tiles))
        return value

    def calculateStrategicValue(self, player):
        """Net worth plus a bonus for each complete color set (placeholder reward)."""
        return self._strategicValueFor(player.cashBalance, player.ownedProperties)

    def dealValueDelta(self, player, deal):
        """How much `player`'s strategic value would change if `deal` happened.

        `deal` must be written from this player's point of view (give = what they hand over).
        """
        before = self.calculateStrategicValue(player)
        owned = [p for p in player.ownedProperties if p not in deal.give_props] + list(deal.get_props)
        after = self._strategicValueFor(player.cashBalance - deal.give_cash + deal.get_cash, owned)
        return after - before

    # ----------------------------------------------------------------- trades
    def validateTrade(self, initiator, target, offerData):
        """Return None if the trade is legal, otherwise a plain-English reason it is not."""
        if initiator.playerName == target.playerName:
            return "You cannot trade with yourself."
        giveProps, recvProps = offerData["giveProps"], offerData["receiveProps"]
        if len(set(giveProps)) != len(giveProps) or len(set(recvProps)) != len(recvProps):
            return "A property is listed twice."
        for propId in giveProps:
            tile = self.board.getTile(propId)
            if not isinstance(tile, PropertyTile) or propId not in initiator.ownedProperties:
                if propId in target.ownedProperties:
                    return (f"{tile.name} is in give=[...], but it belongs to {target.playerName}, so "
                            f"{initiator.playerName} does not own it. To BUY it, list it in get=[...] instead.")
                return (f"{initiator.playerName} does not own {tile.name}, so it cannot be given away. "
                        f"give=[...] must list properties {initiator.playerName} owns.")
            if tile.houseCount > 0:
                return f"{tile.name} has houses and cannot be traded."
        for propId in recvProps:
            tile = self.board.getTile(propId)
            if not isinstance(tile, PropertyTile) or propId not in target.ownedProperties:
                if propId in initiator.ownedProperties:
                    return (f"{tile.name} is in get=[...], but {initiator.playerName} already owns it, so "
                            f"{target.playerName} does not own it. To SELL it, list it in give=[...] instead.")
                return (f"{target.playerName} does not own {tile.name}, so it cannot be received. "
                        f"get=[...] must list properties {target.playerName} owns.")
            if tile.houseCount > 0:
                return f"{tile.name} has houses and cannot be traded."
        if offerData["giveCash"] > initiator.cashBalance:
            return f"{initiator.playerName} only has ${initiator.cashBalance}, not ${offerData['giveCash']}."
        if offerData["receiveCash"] > target.cashBalance:
            return f"{target.playerName} only has ${target.cashBalance}, not ${offerData['receiveCash']}."
        return None

    def proposeTrade(self, initiatorId, targetId, offerData):
        """Original one-step trade (no negotiation). Now validated before it runs."""
        initiator = self.getPlayer(initiatorId)
        target = self.getPlayer(targetId)

        giveNames = [self.board.getTile(pid).name for pid in offerData['giveProps']]
        recvNames = [self.board.getTile(pid).name for pid in offerData['receiveProps']]

        print(f"\n--- TRADE PROPOSAL ---")
        print(f"{initiator.playerName} offers {target.playerName}:")
        print(f"GIVING: ${offerData['giveCash']} and Properties: {giveNames}")
        print(f"FOR: ${offerData['receiveCash']} and Properties: {recvNames}")

        return self.executeTrade(initiator, target, offerData)

    def executeTrade(self, initiator, target, offerData):
        """Run a trade if it is legal. Returns a dict of results, or None if it was refused."""
        problem = self.validateTrade(initiator, target, offerData)
        if problem:
            print(f"  -> TRADE REFUSED: {problem}")
            return None

        # 1. Snapshot pre-trade values (BATNA baseline)
        initiatorPreEquity = self.calculateEquity(initiator)
        targetPreEquity = self.calculateEquity(target)
        initiatorPreStrategic = self.calculateStrategicValue(initiator)
        targetPreStrategic = self.calculateStrategicValue(target)

        # 2. Execute Asset Swap
        initiator.deductFunds(offerData["giveCash"])
        target.addFunds(offerData["giveCash"])
        target.deductFunds(offerData["receiveCash"])
        initiator.addFunds(offerData["receiveCash"])

        for propId in offerData["giveProps"]:
            initiator.removeProperty(propId)
            target.addProperty(propId)
            self.board.getTile(propId).ownerId = target.playerName

        for propId in offerData["receiveProps"]:
            target.removeProperty(propId)
            initiator.addProperty(propId)
            self.board.getTile(propId).ownerId = initiator.playerName

        print("  -> ACTION: Trade Accepted and Executed.")
        self.logEvent(f"{initiator.playerName} and {target.playerName} completed a trade.")

        # 3. Calculate post-trade values and print the RL Reward shift
        initiatorPostEquity = self.calculateEquity(initiator)
        targetPostEquity = self.calculateEquity(target)
        initiatorPostStrategic = self.calculateStrategicValue(initiator)
        targetPostStrategic = self.calculateStrategicValue(target)

        print("\n  -> [BATNA EVALUATOR]:")
        print(f"     {initiator.playerName} Equity Shift: ${initiatorPreEquity} -> ${initiatorPostEquity} (Delta: ${initiatorPostEquity - initiatorPreEquity})")
        print(f"     {target.playerName} Equity Shift: ${targetPreEquity} -> ${targetPostEquity} (Delta: ${targetPostEquity - targetPreEquity})")
        print(f"     Strategic value (with set bonus): {initiator.playerName} {initiatorPostStrategic - initiatorPreStrategic:+d}, "
              f"{target.playerName} {targetPostStrategic - targetPreStrategic:+d}")

        return {
            "initiator": initiator.playerName,
            "target": target.playerName,
            "initiator_equity_delta": initiatorPostEquity - initiatorPreEquity,
            "target_equity_delta": targetPostEquity - targetPreEquity,
            "initiator_strategic_delta": initiatorPostStrategic - initiatorPreStrategic,
            "target_strategic_delta": targetPostStrategic - targetPreStrategic,
        }

    # ------------------------------------------------------------ negotiation
    def negotiationPhase(self, player):
        """After moving, the active player's agent may open a negotiation (MVP: 2 players)."""
        if self.agents.get(player.playerName) is None:
            return
        opponents = [p for p in self.players if p.playerName != player.playerName]
        if len(opponents) != 1:
            raise NotImplementedError("The negotiation MVP supports exactly 2 players.")
        opponent = opponents[0]
        if opponent.playerName not in self.agents:
            return  # nobody on the other side to talk to
        if not any(p.ownedProperties for p in self.players):
            return  # nothing to trade yet; skip so we do not waste model calls

        result = run_negotiation(self, player, opponent, self.maxNegotiationTurns)
        self.negotiationResults.append(result)

    # ----------------------------------------------------------- game state
    def serializeGameState(self, activePlayer):
        def formatProperties(propertyIds):
            props = []
            for propId in propertyIds:
                tile = self.board.getTile(propId)
                props.append({
                    "id": tile.index,
                    "name": tile.name,
                    "colorGroup": tile.colorGroup,
                    "price": tile.price,
                    "currentRent": self.getRent(tile)
                })
            return props

        stateDict = {
            "myState": {
                "name": activePlayer.playerName,
                "cash": activePlayer.cashBalance,
                "position": self.board.getTile(activePlayer.boardPosition).name,
                "ownedProperties": formatProperties(activePlayer.ownedProperties)
            },
            "opponents": []
        }

        for player in self.players:
            if player.playerName != activePlayer.playerName:
                stateDict["opponents"].append({
                    "name": player.playerName,
                    "cash": player.cashBalance,
                    "ownedProperties": formatProperties(player.ownedProperties)
                })

        return json.dumps(stateDict, indent=2)
