class Solution(object):
    def countSmaller(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        results = []
        d = {}
        for i, n in enumerate(nums[::-1]):
            count = 0
            if i == 0:
                results.insert(0, count)
                d[n] = count
                continue
            for j in nums[-i:]:
                if j == n:
                    d[n] += count
                    count = d[n]
                    break
                if j < n:
                    count += 1
            results.insert(0, count)
            d[n] = count
        return results
