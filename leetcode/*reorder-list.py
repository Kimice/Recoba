# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: void Do not return anything, modify head in-place instead.
        """
        i = head
        j = head
        try:
            while i.next.next:
                k = j
                while j.next.next:
                    k = j
                    j = j.next

                j.next.next = i.next
                i.next = j.next
                j.next = None
                i = i.next.next
                j = k
        except:
            pass


a = ListNode('a')
b = ListNode('b')
c = ListNode('c')
d = ListNode('d')
e = ListNode('e')

a.next = b
b.next = c
c.next = d
d.next = e

Solution().reorderList(a)

v = a
while v:
    print v.val
    v = v.next

