# https://leetcode.com/problems/intersection-of-two-arrays/


# Given two arrays, write a function to compute their intersection.
#
# Example:
# Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2].
#
# Note:
# Each element in the result must be unique.
# The result can be in any order.


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution(object):
    def preOrder(self, root, level, l):
        if root:
            if len(l) < level + 1:
                l.append([])
            if level % 2 == 0:
                l[level].append(root.val)
            else:
                l[level].insert(0, root.val)
            self.preOrder(root.left, level + 1, l)
            self.preOrder(root.right, level + 1, l)

    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        l = []
        self.preOrder(root, 0, l)
        return l
