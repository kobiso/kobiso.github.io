---
title: "HackerRank: Is This a Binary Search Tree"
categories:
  - Coding challenge
tags:
  - hackerrank
  - tree
  - binary tree
  - binary search tree
header:
  teaser: /assets/images/hackerrank.png
  overlay_image: /assets/images/hackerrank.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing answer codes of mine about [HackerRank: Is This a Binary Search Tree](https://www.hackerrank.com/challenges/is-binary-search-tree/problem).

{% include toc title="Table of Contents" icon="file-text" %}

# HackerRank: Is this a Binary Search Tree (in Data Structures) 

## Problem Statement
For the purposes of this challenge, we define a binary tree to be a binary search tree with the following ordering requirements:

The data value of every node in a node's left subtree is less than the data value of that node.
The data value of every node in a node's right subtree is greater than the data value of that node.
Given the root node of a binary tree, can you determine if it's also a binary search tree?

Complete the function in your editor below, which has 1 parameter: a pointer to the root of a binary tree.
It must return a boolean denoting whether or not the binary tree is a binary search tree.
You may have to write one or more helper functions to complete this challenge.

## Answer Code using Deque (in Python3) 
- Time complexity: $$O(n)$$ ($$n$$ is the number of nodes)

```python
""" Node is defined as
class node:
  def __init__(self, data):
      self.data = data
      self.left = None
      self.right = None
"""
from collections import deque
import math

def check_binary_search_tree_(root):
    dq = deque([(root, -math.inf, math.inf)])
    while dq:
        node_now, minV, maxV = dq.popleft()
        if node_now == None: continue
        if node_now.data <= minV or node_now.data >= maxV: return False
        if node_now.left: dq.append((node_now.left, minV, node_now.data))
        if node_now.right: dq.append((node_now.right, node_now.data, maxV))
    return True
```

## Answer Code using Recursion (in Python3) 
- Time complexity: $$O(n)$$ ($$n$$ is the number of nodes)

```python
""" Node is defined as
class node:
  def __init__(self, data):
      self.data = data
      self.left = None
      self.right = None
"""
import math
visited = -math.inf

# Inorder traversal using recursive function
def check_binary_search_tree_(root):
    global visited
    if not root: return True
    if not check_binary_search_tree_(root.left): return False
    if root.data <= visited: return False        
    visited = root.data
    if not check_binary_search_tree_(root.right): return False
    return True
```