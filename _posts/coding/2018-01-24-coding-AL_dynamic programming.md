---
title: "Dynamic Programming"
categories:
  - Algorithm
tags:
  - algorithm
  - dynamic programming
header:
  teaser: /assets/images/dynamic programming/coin.png
  overlay_image: /assets/images/dynamic programming/coin.png
  overlay_filter: 0.4
sidebar:
  nav: "cs"
author_profile: false
---

Learn about **Dynamic Programming** which is a famous and important algorithm for solving problems.

{% include toc title="Table of Contents" icon="file-text" %}

Dynamic Programming
===================
 
**Dynamic programming** is a method for solving a complex problem by following this steps:

1. Break it down into a collection of simpler subproblems
2. Solve each of subproblems just once and store their solutions
3. When the same subproblem occurs, look up the previously computed solutions thereby saving computation time and space

- **Memoization**: The technique of storing solutions to subproblems instead of recomputing them.

# Coin Changing Problem
## Problem Statement
For a given set of denominations, find the minimum number of coins with which a given amount of money can be paid.
Assume that you can use as many coins of a particular denomination as necessary.

## Approach with Greedy Algorithm
- **Process**
  - Using greedy algorithm, we can simply choose the largest denomination not exceeding the remaining amount of money to be paid.
  - As long as the remaining amount is greater than zero, the process is repeated.
  
- **Exception**
  - The algorithm may return a suboptimal result.
  - For an amount of 6 and coins of values 1, 3, 4, we get 6 = 4+ 1+ 1, but the optimal solution here is 6 = 3 + 3.

## Approach with Dynamic Programming
A dynamic algorithm finds solutions to this problem for all amounts not exceeding the given amount, and for increasing sets of denominations.
For example, it would consider all the amounts from 0 to 6, and the following sets of denominations: $$\varnothing,\{1\},\{1,3\},\{1,3,4\}$$.
Let $$dp[i, j]$$ be the minimum number of coins needed to pay the amount $$j$$ if we use the set containing the $$i$$ smallest denominations.

The number of coins needed must satisfy the following rules:
  - no coins are needed to pay a zero amount: $$dp[i, 0] = 0$$
  - if there are no denominations and the amount is positive, there is no solution: $$dp[0, j]=\infty$$
  - if the amount to be paid is smaller than the highest denomination $$c_i$$, this denomination can be discarded: $$dp[i,j] = dp[i-1,j]$$
  - otherwise, either we use a coin of the highest denomination and a smaller amount to be paid remains, or we don't use coins of the highest denomination:
  $$dp[i,j] = min(dp[i,j-c_i] + 1, dp[i-1,j])$$

The following table shows all the solutions to sub-problems considered for the example.
![Coin]({{ site.url }}{{ site.baseurl }}/assets/images/dynamic programming/coin.png){: .align-center}{:height="60%" width="60%"}

## Implementation
Consider $$n$$ denominations, $$0 \leq c_0 \leq c_1 \leq ... \leq c_{n-1}$$.
The algorithm processes the respective denominations and calculates the minimum number of coins needed to pay every amount from 0 to $$k$$.

```python
# In Python 3
# Time complexity and space complexity are O(nk)
import math
def dynamic_coin_changing(C, k):
    n = len(C)
    # Create two-dimensional array with all zeros
    dp = [[0]*(k+1) for _ in range(n+1)] # same with [[0]*(k+1)]*(n+1)
    dp[0] = [0] + [math.inf] * k
    for i in range(1, n+1):
        for j in range(C[i-1]):
            dp[i][j] = dp[i-1][j]
        for j in range(C[i-1], k+1):
            dp[i][j] = min(dp[i][j-C[i-1]] + 1, dp[i-1][j])
     return dp[n]    
```

- Space optimized version: during the calculation of $$dp$$, we only use the previous row, so we don't need to remember all of the rows.

```python
# In Python 3
import math
def dynamic_coin_changing(C, k):
    n = len(C)
    dp = [0] + [math.inf] * k
    for i in range(1, n+1):
        for j in range(C[i-1], k+1):
            dp[j] = min(dp[j-C[i-1]] + 1, dp[j])
    return dp
```

# References
- Wekipedia: Dynamic programming [[Link](https://en.wikipedia.org/wiki/Dynamic_programming)]
- Codility: Dynamic programming [[Link](https://codility.com/media/train/15-DynamicProgramming.pdf)]

