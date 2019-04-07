---
title: "LeetCode 2. Add Two Numbers"
categories:
  - Coding challenge
tags:
  - linked_list
  - math
header:
  teaser: /assets/images/leetcode.png
  overlay_image: /assets/images/leetcode.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing an answer code of mine about [2. Add Two Numbers of LeetCode](https://leetcode.com/problems/add-two-numbers/).

{% include toc title="Table of Contents" icon="file-text" %}

# 2. Add Two Numbers of LeetCode
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

**Example:**

```
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

# Answer Code in Python 3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # O(m+n+max(m,n))
    # Runtime: 88 ms, faster than 90.13% of Python3 online submissions for Add Two Numbers.
    # Memory Usage: 13.3 MB, less than 5.21% of Python3 online submissions for Add Two Numbers.
    def listToDigit(self, linked_list):
        
        digit_str=''
        while (not isinstance(linked_list, type(None))):
            digit_str = str(linked_list.val) + digit_str
            linked_list = linked_list.next            
        
        return int(digit_str)

    # O(max(m,n))
    # Runtime: 108 ms, faster than 56.86% of Python3 online submissions for Add Two Numbers.
    # Memory Usage: 13.3 MB, less than 5.21% of Python3 online submissions for Add Two Numbers.
    def sumCarry(self, l1, l2):
        result = []
        carry = False        
        while ((not isinstance(l1, type(None))) or (not isinstance(l2, type(None))) or carry):
            
            if isinstance(l1, type(None)): l1_val = 0
            else: l1_val = l1.val
            if isinstance(l2, type(None)): l2_val = 0
            else: l2_val = l2.val

            if carry: sum_val = l1_val + l2_val + 1
            else: sum_val = l1_val + l2_val
                
            if sum_val >= 10:
                carry = True
                sum_val = sum_val-10
            else: carry = False

            result.append(sum_val)
            if not isinstance(l1, type(None)): l1 = l1.next
            if not isinstance(l2, type(None)): l2 = l2.next
        return result
            
    
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # return list(map(int, str(self.listToDigit(l1) + self.listToDigit(l2))[::-1]))
        return self.sumCarry(l1,l2)
```
