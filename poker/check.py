with open('logs/holdem.log', 'r') as f:
    correct = wrong = draw = p1 = p2 = g1 = g2 = 0
    for i in f.readlines():
        guess_winner, hand, table, winner = map(lambda x: x.strip(), i.split(' | '))
        if guess_winner == 'Kimice':
            g1 += 1
        if guess_winner == 'Hanson':
            g2 += 1
        if winner == 'Kimice':
            p1 += 1
        if winner == 'Hanson':
            p2 += 1
        if winner == 'draw':
            draw += 1
        if guess_winner == winner:
            correct += 1
        else:
            wrong += 1
    print correct, wrong, draw, p1, p2, g1, g2
    print 'correct percent: %f%%' % (float(correct) / (correct + wrong + draw) * 100)
    print 'wrong percent: %f%%' % (float(wrong) / (correct + wrong + draw) * 100)
    print 'draw percent: %f%%' % (float(draw) / (correct + wrong + draw) * 100)

