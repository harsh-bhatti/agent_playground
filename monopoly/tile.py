class Tile:
    def __init__(self, name, index):
        self.name = name
        self.index = index

    def onLand(self, player, gameManager):
        pass

class PropertyTile(Tile):
    def __init__(self, name, index, price, rentLevels, colorGroup):
        super().__init__(name, index)
        self.price = price
        self.rentLevels = rentLevels
        self.colorGroup = colorGroup
        self.ownerId = None
        self.isMortgaged = False
        self.houseCount = 0

    def onLand(self, player, gameManager):
        if self.ownerId is None:
            gameManager.promptBuyDecision(player, self)
        elif self.ownerId != player.playerName and not self.isMortgaged:
            rentDue = gameManager.getRent(self)
            gameManager.processRentPayment(player, self.ownerId, rentDue)
        elif self.ownerId == player.playerName:
            print(f"  -> INFO: {player.playerName} already owns this property.")

    def calculateRent(self):
        return self.rentLevels[self.houseCount]

class TaxTile(Tile):
    def __init__(self, name, index, taxAmount):
        super().__init__(name, index)
        self.taxAmount = taxAmount

    def onLand(self, player, gameManager):
        player.deductFunds(self.taxAmount)
        print(f"  -> TAX: {player.playerName} paid ${self.taxAmount} for {self.name}.")

class ActionTile(Tile):
    def onLand(self, player, gameManager):
        print(f"  -> ACTION: {player.playerName} drew a card on {self.name}.")

class SafeTile(Tile):
    def onLand(self, player, gameManager):
        pass