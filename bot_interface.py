
import sys

class GameObject(object):

    def __init__(self, uid):
        self.uid = uid
        self.posx = 0
        self.posy = 0
        self.velx = 0
        self.vely = 0
        self.radius = 0

    def update(self, **kargs):
        self.__dict__.update(kargs)

class Ship(GameObject):
    def __init__(self, uid):
        GameObject.__init__(self, uid)
        self.ang = 0
        self.velang = 0
        self.charge = 0
        self.score = 0

class Rock(GameObject):
    pass

class Laser(GameObject):
    def __init__(self, uid):
        GameObject.__init__(self, uid)
        self.lifetime = 0
        self.owner = 0

class Action(object):

    def __init__(self, thrust=0, sideThrustFront=0, sideThrustBack=0, shoot=0, **kargs):
        self.thrust = thrust
        self.sideThrustFront = sideThrustFront
        self.sideThrustBack = sideThrustBack
        self.shoot = shoot
        self.__dict__.update(kargs)

    def __add__(self, other):
        action = Action()
        for var in vars(action):
            setattr(action, var, getattr(self, var) + getattr(other, var))
        return action

Action.THRUST = Action(1, 0,0, 0)
Action.TURN_LEFT = Action(0, 1,-1, 0)
Action.TURN_RIGHT = Action(0, -1,1, 0)
Action.STRAFE_LEFT = Action(0, -1,-1, 0)
Action.STRAFE_RIGHT = Action(0, 1,1, 0)
Action.SHOOT = Action(0, 0,0, 1)
Action.IDLE = Action(0, 0,0, 0)


class BotBase(Ship):

    def __init__(self):
        Ship.__init__(self, 0)

    def process(self, gamestate):
        pass

class GameState(object):

    def __init__(self, bot):
        self.ships = {}
        self.rocks = {}
        self.lasers = {}
        self.bot = bot
        self.__missing = set([])
        self.timestep = 0
        self.tick = 0
        self.arenaRadius = 0
        self.__last_action = Action.IDLE

    def __get_ship(self, uid):
        if uid not in self.ships:
            ship = Ship(uid)
            self.ships[uid] = ship
        self.__missing.discard(uid)
        return self.ships[uid]

    def __get_rock(self, uid):
        if uid not in self.rocks:
            rock = Rock(uid)
            self.rocks[uid] = rock
        self.__missing.discard(uid)
        return self.rocks[uid]

    def __get_laser(self, uid):
        if uid not in self.lasers:
            laser = Laser(uid)
            self.lasers[uid] = laser
        self.__missing.discard(uid)
        return self.lasers[uid]

    def log(self, message):
        sys.stderr.write(str(message) + "\r\n")
        sys.stderr.flush()

    def connect(self):

        self.log("Bot Loaded")

        while 1:

            self.__missing |= set(self.ships.keys())
            self.__missing |= set(self.rocks.keys())
            self.__missing |= set(self.lasers.keys())

            for message in sys.stdin.readline().strip().split("|"):
                self.__exec(message)

            sys.stdin.flush()

            for miss in self.__missing:
                self.ships.pop(miss, None)
                self.rocks.pop(miss, None)
                self.lasers.pop(miss, None)
            self.__missing.clear()

            try:
                self.__last_action = self.bot.process(self)
            except Exception as e:
                import traceback
                traceback.print_exc()
                sys.stderr.write("\r\n")
                sys.stderr.flush()

            sys.stdout.write("%f %f %f %i" % (self.__last_action.thrust, self.__last_action.sideThrustFront, self.__last_action.sideThrustBack, self.__last_action.shoot))
            sys.stdout.write("\r\n")
            sys.stdout.flush()


    def __exec(self, message):

        tokens = message.strip().split(" ")

        if tokens[0] == "ship":
            self.__get_ship(int(tokens[1])).update(posx=float(tokens[2]), posy=float(tokens[3]), velx=float(tokens[4]), vely=float(tokens[5]), radius=float(tokens[6]), ang=float(tokens[7]), velang=float(tokens[8]), charge=float(tokens[9]), score=int(tokens[10]))

        if tokens[0] == "rock":
            self.__get_rock(int(tokens[1])).update(posx=float(tokens[2]), posy=float(tokens[3]), velx=float(tokens[4]), vely=float(tokens[5]), radius=float(tokens[6]))
        
        if tokens[0] == "laser":
            self.__get_laser(int(tokens[1])).update(posx=float(tokens[2]), posy=float(tokens[3]), velx=float(tokens[4]), vely=float(tokens[5]), radius=float(tokens[6]), lifetime=float(tokens[7]), owner=int(tokens[8]))

        if tokens[0] == "tick":
            self.tick = int(tokens[1])

        if tokens[0] == "arenaRadius":
            self.arenaRadius = float(tokens[1])

        if tokens[0] == "uid":
            bot_uid = int(tokens[1])
            self.bot.uid = bot_uid
            self.ships[bot_uid] = self.bot
    
        if tokens[0] == "timestep":
            self.timestep = float(tokens[1])







