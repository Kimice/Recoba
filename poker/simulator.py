# from calc.holdem_calc import calculate
from calc.parallel_holdem_calc import calculate
from game.modules import Player, Pitch
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler('logs/holdem.log'))


def guess(p):
    k = int(raw_input('num of players:'))
    guess_winner = p.players[k - 1]
    return guess_winner


def simulator():
    p = Pitch()
    for name in [
        'Kimice',
        'Hanson',
        # 'Vanlow',
        # 'XinLei',
        # 'FuckALl',
        # 'Surround'
    ]:
        p.players.append(Player(name))

    p.shuffle_deck()
    p.deal()
    p.show_table()
    guess_winner = guess(p)
    # for player in p.players:
    #     player.show_hand()
    print '*' * 20
    hole_cards = []
    for player in p.players:
        hole_cards.extend(player.hand)
    # calculate(hole_cards)
    # p.betting_round()
    print '=' * 5, 'flop', '=' * 5
    p.flop()
    p.show_table()
    calculate(hole_cards, board=p.table)
    # p.betting_round()
    print '=' * 5, 'turn', '=' * 5
    p.turn_river()
    p.show_table()
    calculate(hole_cards, board=p.table)
    # p.betting_round()
    print '=' * 5, 'river', '=' * 5
    p.turn_river()
    p.show_table()
    winner_index = calculate(hole_cards, board=p.table).index(1.0)
    winner = p.players[winner_index - 1].name if winner_index != 0 else 'draw'
    logger.info('{} | {} | {} | {}'.format(
        guess_winner.name,
        [{player.name: player.hand} for player in p.players],
        p.table,
        winner
    ))

if __name__ == '__main__':
    simulator()
    # calculate(('Ac', 'Ad', '7h', '7d', 'Ah', '9s'), board=['As', '7s', '7c', '5s'])
