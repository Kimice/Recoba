import random
from collections import deque
from common.errors import NoFundsError


class Game:
    def __init__(self):
        self.pitches = []


class Player:
    player_cnt = 0

    def __init__(self, name):
        self.allin = False
        self.name = name
        self.hand = []
        self.purse = 100
        self.ante = 0
        self.check = False
        self.raize = False
        self.fold = False
        self.uniq = Player.player_cnt
        Player.player_cnt += 1

    def show_hand(self):
        print self.name + ' is holding ' + ' '.join(self.hand)
        # print '\nTheir purse is currently: $' + str(self.purse)
        # print 'Their ante is currently: $' + str(self.ante) + '\n'


class Pitch:
    def __init__(self):
        self.players = deque()
        self.deck = [i + j for j in 'shdc' for i in '23456789TJQKA']
        self.table = []
        self.pot = 0
        self.last_raise = 0
        self.small_blind = 1
        self.big_blind = self.small_blind * 2
        self.sb_position = 0
        self.is_raise = False
        self.guess_player = ''

    def get_random_sb(self):
        self.sb_position = random.choice(range(len(self.players)))

    def betting_round(self):
        while True:
            unfold_players = filter(lambda p: not p.fold, self.players)
            for player in unfold_players:
                print '{}\'s turn. purse:{}, pot:{}'.format(
                    player.name, player.purse, self.pot)
                self.decide(player)
                self.pot += player.ante
            if not filter(lambda x: not (x.allin or x.ante == self.last_raise), self.players):
                break

        for play in self.players:
            self.pot += play.ante
            play.ante = 0
            play.check = False
            play.raize = False
        self.last_raise = 0
        return 'pot is at $' + str(self.pot)

    def call(self, amount, player):
        player.purse -= amount
        player.ante += amount

    def deal(self):
        for _ in range(2):
            for play in self.players:
                play.hand.append(self.deck.pop())

    def decide(self, player):
        try:
            player.show_hand()
            if player.fold or player.allin:
                return
            dif = self.last_raise - player.ante
            if dif >= player.purse:
                player.allin = True
            k = ''
            if dif == 0:
                while k not in ('c', 'r', 'f', 'a'):
                    k = raw_input("[c] to check, [r] to raise, [f] to fold, or [a] to ALL IN: ")
                if k == 'c':
                    print player.name + ' checks'
                    player.check = True
                elif k == 'r':
                    j = int(raw_input("How much? "))
                    self.raize(j, player)
                    print player.name + ' raises $'+str(j)
                elif k == 'f':
                    print player.name + ' folds'
                    self.fold(player)
                elif k == 'a':
                    print player.name + ' allin'
                    self.allin(player)
            else:
                while k not in ('c', 'r', 'f'):
                    if player.allin:
                        k = raw_input("[c] to go ALL IN with $" + str(player.purse) + ", or [f] to fold: ")
                        while k not in ('c', 'f'):
                            k = raw_input("[c] to go ALL IN with $" + str(player.purse) + ", or [f] to fold: ")
                    else:
                        k = raw_input("[c] to call $" + str(dif) + ", [r] to raise, or [f] to fold: ")
                if k == 'c':
                    self.call(dif, player)
                    print player.name + ' calls $' + str(dif)
                elif k == 'r':
                    j = int(raw_input("How much? "))
                    self.raize(j, player)
                    print player.name + ' raises $' + str(j)
                elif k == 'f':
                    print player.name + ' folds'
                    self.fold(player)
        except NoFundsError:
            self.decide(player)

    def finish_hand(self):
        winner = 0
        self.players[winner].purse += self.pot
        self.pot = 0
        for play in self.players:
            play.show_hand()
            play.fold = False
            play.hand = []
        return 'Hand done'

    def flop(self):
        self.deck.pop()
        for i in range(3):
            self.table.append(self.deck.pop())

    def raize(self, amount, player):
        if player.purse - amount < 0:
            print "You don't have the funds\n"
            raise NoFundsError("You don't have the funds")
        else:
            player.purse -= amount
            player.ante += amount
            player.raize = True
            self.last_raise += player.ante

    def allin(self, player):
        player.ante += player.purse
        player.purse = 0
        self.last_raise += player.ante
        player.allin = True

    def fold(self, player):
        for _ in range(2):
            player.hand.pop()
        player.fold = True

    def show_pitch(self):
        print '\nCurrently Playing:'
        for play in self.players:
            print ' ' + play.name + ' '
        print '\n# of Cards Remaining in Deck: ' + str(len(self.deck))
        print ' | Current Pot: $' + str(self.pot) + ' | Last Raise: $' + str(self.last_raise)

    def show_table(self):
        print '\n' + '*' * 10 + 'On the table: ' + ' '.join(self.table) + ' ' + '*' * 10
        for i in self.players:
            i.show_hand()

    def shuffle_deck(self):
        for i in range(random.randint(1, 6)):
            random.shuffle(self.deck)

    def turn_river(self):
        self.deck.pop()
        self.table.append(self.deck.pop())


