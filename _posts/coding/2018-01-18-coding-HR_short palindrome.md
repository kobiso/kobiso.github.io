---
title: "HackerRank: Short Palindrome"
categories:
  - Coding challenge
tags:
  - hackerrank
  - search
header:
  teaser: /assets/images/hackerrank.png
  overlay_image: /assets/images/hackerrank.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing answer codes of mine about [HackerRank: Short Palindrome](https://www.hackerrank.com/challenges/short-palindrome/problem).

{% include toc title="Table of Contents" icon="file-text" %}

# HackerRank: Short Palindrome (in Algorithm) 

## Problem Statement
Consider a string, $$s$$, of $$n$$ lowercase English letters where each character, $$s_i (0\leq i < n)$$, denotes the letter at index $$i$$ in $$s$$.
We define an $$(a,b,c,d)$$ palindromic tuple of $$s$$ to be a sequence of indices in $$s$$ satisfying the following criteria:

- $$s_a = s_d$$, meaning the characters located at indices $$a$$ and $$d$$ are the same.
- $$s_b = s_c$$, meaning the characters located at indices $$b$$ and $$c$$ are the same.
- $$0\leq a< b< c< d< \mid s \mid$$, meaning that $$a, b, c,$$ and $$d$$ are ascending in value and are valid indices within string $$s$$.

Given $$s$$, find and print the number of $$(a,b,c,d)$$ tuples satisfying the above conditions. As this value can be quite large, print it modulo $$10^9+7$$.

## Naive Answer Code (in Python3) 
- Time complexity: $$O(n^3)$$

```python
# Time complexity: O(n^3)
#!/bin/python3

import sys
import string

def shortPalindrome(s):
    mod = pow(10,9)+7
    cnt = 0

    for i in range(len(s)):
        for j in range(i+1, len(s)):
            for k in range(j+1, len(s)):
                if s[j]==s[k]:
                    cnt += s[k+1:].count(s[i])

    return cnt % mod

if __name__ == "__main__":
    s = input().strip()
    result = shortPalindrome(s)
    print(result)
```

## Final Answer Code (in Python3) 
- Time complexity: $$O(n)$$

```python
# Time complexity: O(n)
#!/bin/python3

import sys
import string

def shortPalindrome(input_string):
    c_dic = dict(zip(string.ascii_lowercase, range(26)))  # ascii value dict for every alphabet
    ip_ascii = [c_dic[i] for i in input_string]  # change input as ascii value
    one_c = [0] * 26  # for one character occurred (e.g. a)
    two_c = [[0] * 26 for _ in range(26)]  # for two chracter occurred (e.g. ab)
    thr_c = [0] * 26  # for three chracter occured following the rule (e.g. abb)
    total = 0

    for current in ip_ascii:
        # sum the number of matching palindrome when the last character is 'current' (e.g. abba)
        total += thr_c[current]
        for i in range(26):  # for every alphabet, sum the number of sequence such as 'abb'
            thr_c[i] += two_c[i][current]
        for i in range(26):  # for every alphabet, sum the number of sequence such as 'ab'
            two_c[i][current] += one_c[i]
        one_c[current] += 1  # Count the occurred character

    return total % (10 ** 9 + 7)

if __name__ == "__main__":
    s = input().strip()
    result = shortPalindrome(s)
    print(result)

```