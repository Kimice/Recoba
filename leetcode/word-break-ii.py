# https://leetcode.com/problems/word-break-ii/


# Given a string s and a dictionary of words dict,
# add spaces in s to construct a sentence where each word is a valid dictionary word.
#
# Return all such possible sentences.
#
# For example, given
# s = "catsanddog",
# dict = ["cat", "cats", "and", "sand", "dog"].
#
# A solution is ["cats and dog", "cat sand dog"].


import functools


def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer


class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: List[str]
        """

        @memoize
        def _wordBreak(s):
            results = []
            for word in wordDict:
                if s == word:
                    results.append(word)
                elif s.startswith(word):
                    for result in _wordBreak(s[len(word):]):
                        results.append(word + ' ' + result)

            return results

        return _wordBreak(s)
