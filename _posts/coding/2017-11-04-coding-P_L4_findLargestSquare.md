---
title: "Programmers Level4. Find Largest Square"
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

Sharing answer codes of mine about [Programmers Level4. findLargestSquare](https://programmers.co.kr/learn/challenge_codes/187).

{% include toc title="Table of Contents" icon="file-text" %}

# Level 4: Find Largest Square

## Problem Statement
Given a M x N matrix filled with 'O' and 'X', find the width of largest square sub-matrix of 'O''s present in it.
For example,

**Input:**<br/>
X O O O X <br/>
X O O O O <br/>
X X O O O <br/>
X X O O O <br/>
X X X X X <br/>

**Output:**<br/>
The width of largest square sub-matrix of 'O''s is 9

## Answer code (in C++) 

```cpp
#include<iostream>
#include<vector>
#include<utility>
using namespace std;

// Solved the problem by Dynamic Programming
int findLargestSquare(vector<vector<char>> board)
{
  int answer = 0;
  
  vector<vector<int>> fls(board.size(), vector<int>(board[0].size(), 0));
    
  for (vector<int>::size_type i=0; i<board.size(); ++i){ // Construting a new matrix for scores
    for (vector<int>::size_type j=0; j<board[0].size(); ++j){
      if (board[i][j] == 'O'){
        fls[i][j] = 1;
      }
      else{
        fls[i][j] = 0;
      }
    } 
  }
  
  for (vector<int>::size_type i=1; i<fls.size(); ++i){
    for (vector<int>::size_type j=1; j<fls[0].size(); ++j){
      if (fls[i][j] == 1){  // Solve current subproblem by using the result of previous subproblem
        fls[i][j] = min(min(fls[i-1][j],fls[i][j-1]),fls[i-1][j-1])+1; // Check left, up, up-left
      }
      if (answer < fls[i][j]){ // Update the Max
        answer = fls[i][j];
      }
    } 
  }

  return answer*answer; // Calculate the width
}
int main()
{
  
  vector<vector<char>> board{
        {'X','O','O','O','X'},
        {'X','O','O','O','O'},
        {'X','X','O','O','O'},
        {'X','X','O','O','O'},
        {'X','X','X','X','X'}};

  cout<<findLargestSquare(board);
}

```

## Dynamic Programming & Memoization
- **Dynamic Programming (DP)**
  - Solve a problem by finding the overlapping subproblems and then cache those results for future recursive calls
  - Sometimes, top-down dynamic programming is called as **"Memoization"**  
  - Examples of DP: Fibonacci Numbers, Longest Common Subsequence (LCS), Knapsack Problem, Matrix Product Parenthesization