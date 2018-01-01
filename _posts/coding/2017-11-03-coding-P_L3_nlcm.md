---
title: "Programmers Level3. nlcm"
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

Sharing answer codes of mine about [Programmers Level3. nlcm](https://programmers.co.kr/learn/challenge_codes/152).

{% include toc title="Table of Contents" icon="file-text" %}

# Level 3: N Least Common Multiple

## Problem Statement
Least Common Multiple (LCM) means the smallest common number among the multiple of two numbers.
For example, LCM of 2 and 7 would be 14.
By extending the definition, the NLCM is the LCM of N numbers.
For the input of N numbers through the nlcm function, return LCM of N numbers.
For example, if [2,6,8,14] is entered, return 168.

## Answer code (in C++) 

```cpp
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

// Great Common Divisor between two by using Euclidean algorithm
long long gcd(long long A, long long B)
{
  return (A%B == 0 ? B : gcd(B, A%B));
}

long long lcm(long long A, long long B) // Least Common Multiple between two
{
  return A * B / gcd(A,B);
}

long long nlcm(vector<int> num)
{
  long long answer;
  sort (num.begin(), num.end()); // Sort it in ascending order

  answer = num[0];
  for (vector<int>::size_type i = 1; i<num.size(); ++i){
    answer = lcm(num[i], answer);
  }

    return answer;
}

int main()
{
    vector<int> test{2,6,8,14};

    cout << nlcm(test);
}

# Compile time: 3ms
```
