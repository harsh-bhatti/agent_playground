class Player:
    def __init__(self, playerName, startingCash=1500):
        self.playerName = playerName
        self.cashBalance = startingCash
        self.boardPosition = 0
        self.ownedProperties = []
        self.inJail = False
        self.jailTurns = 0

    def addFunds(self, amount):
        self.cashBalance += amount

    def deductFunds(self, amount):
        if self.cashBalance >= amount:
            self.cashBalance -= amount
            return True
        return False 

    def moveSpaces(self, steps):
        self.boardPosition += steps
        if self.boardPosition >= 40:
            self.boardPosition %= 40
            self.addFunds(200)
            print(f"  -> INFO: {self.playerName} passed GO and collected $200!")

    def addProperty(self, propertyId):
        self.ownedProperties.append(propertyId)

    def removeProperty(self, propertyId):
        if propertyId in self.ownedProperties:
            self.ownedProperties.remove(propertyId)