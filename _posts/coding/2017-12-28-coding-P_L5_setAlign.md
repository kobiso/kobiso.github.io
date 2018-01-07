---
title: "Programmers Level5. Set Align"
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

Sharing answer codes of mine about [Programmers Level5. Set Align](https://programmers.co.kr/learn/challenge_codes/159).

{% include toc title="Table of Contents" icon="file-text" %}

# Level 5: Set Align

## Problem Statement
Given a natural number N for the number of people, there are N! ways to line up. 
Find the Kth way of lining up people when we sort them alphabetically.
For example, there are six ways to line up for three people.

- 1st way: [1,2,3]
- 2nd way: [1,3,2]
- 3rd way: [2,1,3]
- 4th way: [2,3,1]
- 5th way: [3,1,2]
- 6th way: [3,2,1]

**Input:**<br/>
N = 3, K=5

**Output:**<br/>
Answer = [3,1,2]

## Answer Code (in C++) 

```cpp
#include<iostream>
#include<vector>
using namespace std;

vector<int> setAlign(int n, long long k)
{
  vector<int> answer;
  
  // Make factorial and sequential vectors
  vector<long long> fac(1,0);
  vector<int> seq(1,0);
  long long mul = 1;
    
  for (int i=1; i<=n; ++i){
    seq.push_back(i);
    mul *= i;
    fac.push_back(mul);
    //cout << seq[i] << '/' << fac[i] <<endl;
  }
  
  int num = n;
  long long ki = k, mod, rem;    
  while (num>1){
    --num;
    mod = ki / fac[num];
    rem = ki % fac[num];
    if (rem != 0){
      answer.push_back(*(seq.begin()+(mod+1)));
      seq.erase(seq.begin()+(mod+1));
      ki = rem;
    }
    else{
      answer.push_back(*(seq.begin()+mod));
      seq.erase(seq.begin()+mod);
      if(num==1){
        answer.push_back(*(seq.begin()+1));
      }
      ki = fac[num];
    }
  }
  return answer;
}

int main()
{
  int testn = 4;
  long long testcnt = 24;
  vector<int> testAnswer = setAlign(testn,testcnt);

  for(int i=0; i< testAnswer.size(); i++)
  {
    cout << testAnswer[i] << " ";
  }
}
```

## Answer Code using Algorithm Header(in C++) 

```cpp
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

vector<int> setAlign(int n, long long cnt)
{
  vector<int> answer;
  for(int i=1;i<=n; i++) answer.push_back(i);

  long long i=1;
  do {
    if(cnt==i++) break;
  }while(next_permutation(answer.begin(),answer.end()));

  return answer;
}
int main()
{
  int testn = 4;
  long long testcnt = 6;
  vector<int> testAnswer = setAlign(testn,testcnt);
  
  for(int i=0; i< testAnswer.size(); i++)
  {
    cout << testAnswer[i] << " ";
  }
}
```