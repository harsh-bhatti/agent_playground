import random
import json
from player import Player
from board import Board
from tile import PropertyTile
from actions import EVERYONE
from negotiation import run_negotiation, ask_buy_decision

# PLACEHOLDER for the research reward: how much extra "strategic value" a complete color
# set is worth, as a fraction of the set's total price. calculateEquity() (cash + prices)
# stays as the team's plain net-worth number; this one also credits set completion, which is
# what makes trading for a set worthwhile. We will replace it with the real reward later.
SET_BONUS_RATE = 0.6


class GameManager:
    def __init__(self, playerNames, agents=None, maxNegotiationTurns=6, showThoughts=False,
                 maxCallsPerTurn=None):
        self.players = [Player(name) for name in playerNames]
        self.board = Board()
        self.activePlayerIndex = 0

        # agents: {"Agent A": <agent>, ...}. A player with no agent plays the old automatic way.
        self.agents = agents or {}
        self.maxNegotiationTurns = maxNegotiationTurns
        self.showThoughts = showThoughts

        # Budget of agent calls per dice roll (buy decision + negotiation, retries included).
        # None or 0 means no limit. Each agent reply counts, whether it is an LLM or a rule bot.
        self.maxCallsPerTurn = maxCallsPerTurn
        self.callsThisTurn = 0

        # Properties promised in open offers: {property id: (owner, recipient)}.
        self.locks = {}

        self.turnCount = 0
        self.events = []  # short plain-English history, shown to agents
        self.decisionLog = []  # every agent decision, saved for research
        self.negotiationResults = []

        # How the game ended. endReason stays None while the game is running.
        self.winner = None  # player name, or None for a draw / unfinished game
        self.endReason = None  # "bankruptcy" | "turn_limit" | "stopped_early"
        self.bankruptcies = []  # one dict per eliminated player

    # ------------------------------------------------------------------ basics
    def rollDice(self):
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        return die1 + die2, die1 == die2

    def getPlayer(self, name):
        return next(p for p in self.players if p.playerName == name)

    def logEvent(self, text):
        self.events.append(f"Turn {self.turnCount}: {text}")

    def activePlayers(self):
        """Players still in the game (not bankrupt)."""
        return [p for p in self.players if not p.isBankrupt]

    def isGameOver(self):
        return self.endReason is not None

    def executeTurn(self):
        if self.isGameOver():
            return  # nothing happens after the game has ended
        self.turnCount += 1
        self.callsThisTurn = 0  # a fresh call budget for every dice roll
        activePlayer = self.players[self.activePlayerIndex]
        rollTotal, isDouble = self.rollDice()

        print(f"\n{activePlayer.playerName} rolled a {rollTotal} {'(Doubles!)' if isDouble else ''}")
        self.resolveMovement(activePlayer, rollTotal)
        if not activePlayer.isBankrupt and not self.isGameOver():
            self.negotiationPhase(activePlayer)

        # Doubles roll again, unless the player was just eliminated.
        if not isDouble or activePlayer.isBankrupt:
            self._advanceToNextPlayer()

    def useAgentCall(self):
        """Count one agent call against this turn's budget. Returns False if it is used up."""
        if self.maxCallsPerTurn and self.callsThisTurn >= self.maxCallsPerTurn:
            return False
        self.callsThisTurn += 1
        return True

    def _advanceToNextPlayer(self):
        """Move to the next player who is still in the game."""
        for _ in range(len(self.players)):
            self.activePlayerIndex = (self.activePlayerIndex + 1) % len(self.players)
            if not self.players[self.activePlayerIndex].isBankrupt:
                return

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
        self.chargePlayer(payingPlayer, amount, receivingPlayerId, "rent")

    # ------------------------------------------------- payments & bankruptcy
    def chargePlayer(self, player, amount, creditorName=None, reason=""):
        """Make `player` pay `amount` to another player (creditorName) or to the bank (None).

        This is the one place rent and tax are charged. If the player cannot pay in full,
        they go bankrupt. Returns True if the payment was made.
        """
        payTo = creditorName or "the bank"
        if player.deductFunds(amount):
            if creditorName:
                self.getPlayer(creditorName).addFunds(amount)
            print(f"  -> TRANSACTION: {player.playerName} paid ${amount} {reason} to {payTo}.")
            self.logEvent(f"{player.playerName} paid ${amount} {reason} to {payTo}.")
            return True
        self.declareBankruptcy(player, amount, creditorName, reason)
        return False

    def declareBankruptcy(self, player, owed, creditorName=None, reason=""):
        """Eliminate a player who cannot pay what they owe.

        Owed to a player: that player gets all the bankrupt player's cash and properties.
        Owed to the bank: the cash is lost and the properties go back to the bank, unowned.
        """
        creditor = self.getPlayer(creditorName) if creditorName else None
        paid = player.cashBalance
        player.cashBalance = 0
        if creditor:
            creditor.addFunds(paid)

        handedOver = list(player.ownedProperties)
        for propId in handedOver:
            tile = self.board.getTile(propId)
            player.removeProperty(propId)
            if creditor:
                tile.ownerId = creditor.playerName
                creditor.addProperty(propId)
            else:
                tile.ownerId = None
                tile.houseCount = 0
                tile.isMortgaged = False
        player.isBankrupt = True

        payTo = creditorName or "the bank"
        propNames = ", ".join(self.board.getTile(i).name for i in handedOver) or "none"
        print(f"  -> BANKRUPTCY: {player.playerName} owes ${owed} {reason} but has only ${paid}. "
              f"{player.playerName} is out. {payTo} receives ${paid if creditor else 0} "
              f"and properties: {propNames if creditor else 'none (returned to bank)'}.")
        self.logEvent(f"{player.playerName} could not pay ${owed} {reason} to {payTo} and went bankrupt.")
        self.bankruptcies.append({
            "turn": self.turnCount,
            "player": player.playerName,
            "owed": owed,
            "reason": reason,
            "creditor": creditorName,  # None means the bank
            "cash_handed_over": paid,
            "properties": [self.board.getTile(i).name for i in handedOver],
        })

        remaining = self.activePlayers()
        if len(remaining) == 1:
            self._endGame(remaining[0].playerName, "bankruptcy")

    def finishByTurnLimit(self):
        """The turn limit was reached: highest net worth wins; a tie is a draw (winner None)."""
        if self.isGameOver():
            return
        worths = {p.playerName: self.calculateEquity(p) for p in self.activePlayers()}
        best = max(worths.values())
        leaders = [name for name, worth in worths.items() if worth == best]
        self._endGame(leaders[0] if len(leaders) == 1 else None, "turn_limit")

    def markStoppedEarly(self):
        """The run was cut short (e.g. the model API failed): no winner is declared."""
        if not self.isGameOver():
            self._endGame(None, "stopped_early")

    def _endGame(self, winner, reason):
        self.winner = winner
        self.endReason = reason
        result = f"{winner} wins" if winner else "no winner"
        print(f"\n*** GAME OVER ({reason}): {result} ***")
        self.logEvent(f"Game over ({reason}): {result}.")

    def gameResult(self):
        """How the game ended, for the summary and the saved log."""
        return {
            "winner": self.winner,
            "end_reason": self.endReason,
            "bankruptcies": list(self.bankruptcies),
        }

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
    def tradePartners(self, player):
        """Players `player` may negotiate with: still in the game and controlled by an agent."""
        return [p for p in self.activePlayers()
                if p.playerName != player.playerName and p.playerName in self.agents]

    def findPlayer(self, text):
        """Match a name an agent wrote ("Agent C", "agent c" or just "C") to a player, or None."""
        wanted = (text or "").strip().lower()
        matches = [p for p in self.players if p.playerName.lower() == wanted]
        if not matches:  # allow a short form such as "C" for "Agent C"
            matches = [p for p in self.players if p.playerName.lower().split()[-1] == wanted]
        return matches[0] if len(matches) == 1 else None

    def validateIdea(self, proposer, deal):
        """Check an idea floated to everyone. Returns None if it is fine, otherwise the reason.

        deal is from the proposer's point of view: give_* = what they would hand over,
        get_props = properties they want (each owned by some other player who can trade).
        """
        for propId in deal.give_props:
            tile = self.board.getTile(propId)
            if propId not in proposer.ownedProperties:
                return f"You do not own {tile.name}, so you cannot offer it in give=[...]."
            if tile.houseCount > 0:
                return f"{tile.name} has houses and cannot be traded."
        if deal.give_cash > proposer.cashBalance:
            return f"You only have ${proposer.cashBalance}, not ${deal.give_cash}."
        if not deal.get_props and not deal.get_cash:
            return "An idea must say what you want: list properties in want=[...] or ask for cash with get_cash=M."
        partners = [p.playerName for p in self.tradePartners(proposer)]
        for propId in deal.get_props:
            tile = self.board.getTile(propId)
            if tile.ownerId is None:
                return (f"{tile.name} is unowned (the bank has it), so nobody can trade it. "
                        f"The only way to get it is to land on it and buy it.")
            if tile.ownerId == proposer.playerName:
                return f"You already own {tile.name}. To offer it, list it in give=[...] instead."
            if tile.ownerId not in partners:
                return f"{tile.name} belongs to {tile.ownerId}, who cannot trade."
        return None

    # ------------------------------------------------------------------ locks
    # A property a player has put in an open offer is "promised" to that offer's recipient.
    # It may not be promised to anyone else until that offer closes. An idea floated to
    # everyone promises to EVERYONE, which is compatible with any single recipient.
    def lockConflict(self, owner, recipient, propIds):
        """None if `owner` may promise these properties to `recipient`, otherwise the reason."""
        for propId in propIds:
            held = self.locks.get(propId)
            if held is None or held == (owner, recipient) or held == (owner, EVERYONE):
                continue
            who = "everyone (in an open idea)" if held[1] == EVERYONE else held[1]
            return (f"{self.board.getTile(propId).name} is already promised to {who} in an open "
                    f"offer, so it cannot be offered to anyone else until that offer closes.")
        return None

    def lockOffer(self, owner, recipient, propIds):
        """Record `owner`'s open offer to `recipient` (replacing their earlier one to them)."""
        self.releaseLocks(owner, recipient)
        for propId in propIds:
            self.locks[propId] = (owner, recipient)

    def releaseLocks(self, owner=None, recipient=None):
        """Release locks matching owner and/or recipient; with no arguments, release all."""
        self.locks = {
            propId: held for propId, held in self.locks.items()
            if not ((owner is None or held[0] == owner) and (recipient is None or held[1] == recipient))
        }

    def resolveTradeTarget(self, speaker, targetText, partner=None):
        """Work out who a [SUBMIT_DEAL] is for. Returns (player, None) or (None, reason).

        partner: the player already in this negotiation, or None when opening one.
        Inside a negotiation to=[...] may be left out, but if given it must name the partner.
        When opening, to=[...] may be left out only if there is exactly one possible partner.
        """
        partners = self.tradePartners(speaker)
        options = ", ".join(p.playerName for p in partners) or "nobody"
        if targetText is None:
            if partner is not None:
                return partner, None
            if len(partners) == 1:
                return partners[0], None
            return None, (f"Say who this offer is for: add to=[Player] to the [SUBMIT_DEAL] line. "
                          f"You can trade with: {options}.")

        target = self.findPlayer(targetText)
        if target is None:
            return None, f"There is no player called '{targetText}'. You can trade with: {options}."

        if target.playerName == speaker.playerName:
            return None, f"You cannot make an offer to yourself. You can trade with: {options}."
        if partner is not None and target.playerName != partner.playerName:
            return None, (f"You are negotiating with {partner.playerName}, so to=[...] must be "
                          f"{partner.playerName} (or left out).")
        if target.isBankrupt:
            return None, f"{target.playerName} is bankrupt and out of the game. You can trade with: {options}."
        if target not in partners:
            return None, f"{target.playerName} cannot negotiate. You can trade with: {options}."
        return target, None

    def validateTrade(self, initiator, target, offerData):
        """Return None if the trade is legal, otherwise a plain-English reason it is not."""
        if initiator.playerName == target.playerName:
            return "You cannot trade with yourself."
        for p in (initiator, target):
            if p.isBankrupt:
                return f"{p.playerName} is bankrupt and out of the game, so cannot trade."
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
                if isinstance(tile, PropertyTile) and tile.ownerId is None:
                    return (f"{tile.name} is unowned (the bank has it), so nobody can trade it. "
                            f"The only way to get it is to land on it and buy it.")
                if isinstance(tile, PropertyTile) and tile.ownerId:
                    return (f"{tile.name} belongs to {tile.ownerId}, not {target.playerName}. To ask for it, "
                            f"make the offer to {tile.ownerId} with to=[{tile.ownerId}].")
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

    def _transferCash(self, payer, payee, amount):
        """Move cash between players. Raises instead of failing silently."""
        if amount < 0:
            raise ValueError(f"Cannot transfer a negative amount (${amount}).")
        if not payer.deductFunds(amount):
            raise RuntimeError(f"{payer.playerName} cannot pay ${amount} (has ${payer.cashBalance}).")
        payee.addFunds(amount)

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

        # 2. Execute Asset Swap. Check both cash amounts BEFORE changing anything, so a trade
        #    either happens completely or not at all (no silent half-trades).
        if initiator.cashBalance < offerData["giveCash"] or target.cashBalance < offerData["receiveCash"]:
            raise RuntimeError("Trade passed validation but a player cannot cover the cash.")
        self._transferCash(initiator, target, offerData["giveCash"])
        self._transferCash(target, initiator, offerData["receiveCash"])

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
        """After moving, the active player's agent may open ONE negotiation.

        Only the active player may open. Its opening [SUBMIT_DEAL] names the partner with
        to=[...]; the engine checks that partner before anything else happens.
        """
        if self.agents.get(player.playerName) is None:
            return
        if not self.tradePartners(player):
            return  # nobody (with an agent) left to talk to
        if not any(p.ownedProperties for p in self.players):
            return  # nothing to trade yet; skip so we do not waste model calls

        result = run_negotiation(self, player, None, self.maxNegotiationTurns)
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
