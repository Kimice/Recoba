# -*- coding:utf -*-

import sys
import random

CLUBS = [u'拜仁慕尼黑', u'多特蒙德',
         u'皇家马德里', u'马德里竞技', u'巴萨罗那',
         u'AC米兰', u'国际米兰', u'尤文图斯', u'罗马',
         u'阿森纳', u'曼联', u'曼城', u'利物浦', u'切尔西',
         u'巴黎圣日耳曼', u'波尔图']

NATIONALS = [u'🇨🇳', u'🇧🇷', u'🇦🇷', u'🇺🇾',
             u'🇬🇧', u'🇫🇷', u'🇩🇪', u'🇮🇹',
             u'🇳🇱', u'🇵🇹', u'🇪🇸', u'🇨🇲',
             u'🇯🇵', u'🇰🇷', u'🇨🇮', u'🇳🇬']


def lets_go(k=16, club_or_national=True):
    _team_list = CLUBS if club_or_national else NATIONALS
    random.shuffle(_team_list)
    teams = _team_list[:k]
    print u'⚽ ️⚽ ️⚽ ️对阵表：'
    i = 0
    while i + 2 <= len(teams):
        sys.stdout.write(u'{0} vs {1}\n'.format(teams[i], teams[i+1]))
        i += 2
    sys.stdout.flush()


if __name__ == '__main__':
    team_nums = int(sys.argv[1])
    is_club_or_national = bool(int(sys.argv[2]))
    lets_go(team_nums, is_club_or_national)

