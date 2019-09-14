# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:07:51 2019

@author: midhununnikrishnan
"""

from random import random

def play(player, strat, board):
    sumstrat = sum(strat)
    if sumstrat == 0:
        return -1 # draw
    q = random()
    j = 0
    for i in strat:
       q -= i/sumstrat
       if q > 0:
           j+=1
       else:
           break
    board[j] = player
    return j

def check(board):
    if board[2] == board[4] and board[4] == board[6] and board[6] != 0:
        return board[6]
        
    states = [0,0,0,0]
    flags  = [0,0,0,0]
    for i in range(0,9):
        for j in [0,2,3]:
            if i>j and board[i]==board[i-j-1] and board[i] != 0:
                if board[i] != flags[j]:
                    states[j] = 0
                states[j] += 1
                flags[j] = board[i]
        
        if i%3 == 0:
            states[0] = 0
            
        if max(states) == 2:
            break;
            
    for j in [0,2,3]:
        if states[j]==2:
            return flags[j]
    else:
        return 0

def run_game(strat_map):
    board     = [0]*9
    boardnum  = 0
    player    = 1
    boardnums = [0]
    choices   = []
    while check(board)==0:
        if boardnum >= len(strat_map):
            print(boardnum)
        strat = strat_map[boardnum]
        pos = play(player,strat,board)
        if pos == -1:
            return [boardnums,choices, -1]
        boardnum += player*(3**pos)
        boardnums.append(boardnum)
        choices.append(pos)
        player = (1 if player == 2 else 2)
    return [boardnums,choices,check(board)]

# simulate away
strat_map = []
for j in range(3**9):
    d = [0]*9
    for i in range(9):
        if j%3 == 0:
            d[i] = 1
        j //= 3
    strat_map.append(d)

nrun = 10000
print('training on %d samples.' % nrun );
for iter in range(nrun):
    [bnums,chos,result] = run_game(strat_map)
    if result == -1:
        continue
    
    # strategy changed through reinforcement
    i,j = 1,1
    
    for bnum,cho in zip(bnums,chos):
        strat_map[bnum][cho] += j/27 if i == result else -j/27
        strat_map[bnum][cho] = max(0,strat_map[bnum][cho])
        i = 1 if i == 2 else 1
        j += 1
            
print('training complete.');
