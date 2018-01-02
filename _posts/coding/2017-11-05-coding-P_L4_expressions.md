---
title: "Programmers Level4. Expressions of a Number"
categories:
  - Coding challenge
tags:
  - Programmers
header:
  teaser: /assets/images/programmers.png
  overlay_image: /assets/images/programmers.png
  overlay_filter: 0.4
sidebar:
  nav: "coding"
author_profile: false
---

Sharing answer codes of mine about [Programmers Level4. expressions](https://programmers.co.kr/learn/challenge_codes/156).

{% include toc title="Table of Contents" icon="file-text" %}

# Level 4: Expressions of a Number

## Problem Statement
Given a natural number N, find the number of ways to represent the number N as a sum of consecutive natural numbers.
For example,

**Input:**<br/>
N = 15

**Output:**<br/>
Answer = 4,

since possible ways to represent the number N are $$(1+2+3+4+5), (4+5+6), (7+8), (15)$$.

## Answer code (in C++) 

```cpp
#include<iostream>
using namespace std;
int expressions(int testCase)
{
  int answer = 0;  
  // Loop until half of the testCase, it is unnecessary after then
  for (int i=1; i<=testCase/2; ++i){
    int sum = 0;    
    for (int j=i; j<=testCase; ++j){
      sum += j;
      // If sum is equal to the testCase
      if (sum == testCase){ 
        answer += 1;
        break;
      }
      // Break if sum is bigger than the testCase
      else if (sum > testCase){ 
        break;
      }
    }
  }
  // Plus one to include the testCase itself
  return ++answer;
}

int main()
{
  int testNo = 15;
  int testAnswer = expressions(testNo);

  cout<<testAnswer;
}
```
