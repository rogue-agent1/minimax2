#!/usr/bin/env python3
"""Minimax with iterative deepening, alpha-beta, transposition table."""
import sys, time

class TTable:
    def __init__(self): self.table = {}
    def get(self, key): return self.table.get(key)
    def put(self, key, depth, score, flag):
        self.table[key] = (depth, score, flag)

def minimax_id(evaluate, get_moves, apply_move, hash_state, is_terminal, state, max_time=1.0):
    tt = TTable(); best_move = None; start = time.monotonic()
    
    def search(state, depth, alpha, beta, maximizing):
        if time.monotonic() - start > max_time: raise TimeoutError
        key = hash_state(state)
        cached = tt.get(key)
        if cached and cached[0] >= depth:
            _, score, flag = cached
            if flag == 'exact': return score
            if flag == 'lower': alpha = max(alpha, score)
            elif flag == 'upper': beta = min(beta, score)
            if alpha >= beta: return score
        if depth == 0 or is_terminal(state): return evaluate(state)
        if maximizing:
            val = -float('inf')
            for move in get_moves(state):
                val = max(val, search(apply_move(state, move), depth-1, alpha, beta, False))
                alpha = max(alpha, val)
                if alpha >= beta: break
            flag = 'exact' if val > alpha else 'upper'
        else:
            val = float('inf')
            for move in get_moves(state):
                val = min(val, search(apply_move(state, move), depth-1, alpha, beta, True))
                beta = min(beta, val)
                if alpha >= beta: break
            flag = 'exact' if val < beta else 'lower'
        tt.put(key, depth, val, flag); return val
    
    for depth in range(1, 100):
        try:
            best_score = -float('inf')
            for move in get_moves(state):
                score = search(apply_move(state, move), depth-1, -float('inf'), float('inf'), False)
                if score > best_score: best_score = score; best_move = move
        except TimeoutError: break
    return best_move, best_score, depth-1

if __name__ == "__main__":
    # Nim game demo
    def evaluate(n): return 1 if n % 4 != 0 else -1
    def get_moves(n): return [i for i in [1,2,3] if i <= n]
    def apply_move(n, m): return n - m
    def hash_state(n): return n
    def is_terminal(n): return n == 0
    move, score, depth = minimax_id(evaluate, get_moves, apply_move, hash_state, is_terminal, 15, 0.5)
    print(f"Nim(15): take {move}, score={score}, searched to depth {depth}")
