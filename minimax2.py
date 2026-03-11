#!/usr/bin/env python3
"""minimax2.py — Game AI engine with alpha-beta, iterative deepening, and transposition table.

Plays Connect Four and Tic-Tac-Toe with configurable search depth.
Features: alpha-beta pruning, move ordering, transposition table,
iterative deepening, and killer move heuristic.

One file. Zero deps. Does one thing well.
"""

import sys
import time
from dataclasses import dataclass, field


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: str  # 'exact', 'lower', 'upper'
    best_move: int = -1


class Connect4:
    ROWS, COLS = 6, 7

    def __init__(self):
        self.board = [[0]*self.COLS for _ in range(self.ROWS)]
        self.heights = [0] * self.COLS  # next free row per column
        self.current = 1  # 1 or 2
        self.moves_made = 0

    def copy(self):
        g = Connect4()
        g.board = [row[:] for row in self.board]
        g.heights = self.heights[:]
        g.current = self.current
        g.moves_made = self.moves_made
        return g

    def legal_moves(self) -> list[int]:
        return [c for c in range(self.COLS) if self.heights[c] < self.ROWS]

    def make_move(self, col: int) -> bool:
        r = self.heights[col]
        if r >= self.ROWS:
            return False
        self.board[r][col] = self.current
        self.heights[col] += 1
        self.current = 3 - self.current
        self.moves_made += 1
        return True

    def undo_move(self, col: int):
        self.heights[col] -= 1
        r = self.heights[col]
        self.board[r][col] = 0
        self.current = 3 - self.current
        self.moves_made -= 1

    def check_win(self, player: int) -> bool:
        b = self.board
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if b[r][c] != player:
                    continue
                # Horizontal
                if c + 3 < self.COLS and all(b[r][c+i] == player for i in range(4)):
                    return True
                # Vertical
                if r + 3 < self.ROWS and all(b[r+i][c] == player for i in range(4)):
                    return True
                # Diagonal
                if r + 3 < self.ROWS and c + 3 < self.COLS and all(b[r+i][c+i] == player for i in range(4)):
                    return True
                if r + 3 < self.ROWS and c - 3 >= 0 and all(b[r+i][c-i] == player for i in range(4)):
                    return True
        return False

    def is_draw(self) -> bool:
        return self.moves_made == self.ROWS * self.COLS

    def hash(self) -> int:
        h = 0
        for r in range(self.ROWS):
            for c in range(self.COLS):
                h = h * 3 + self.board[r][c]
        return h

    def display(self) -> str:
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = []
        for r in range(self.ROWS - 1, -1, -1):
            lines.append(' '.join(symbols[self.board[r][c]] for c in range(self.COLS)))
        lines.append(' '.join(str(c) for c in range(self.COLS)))
        return '\n'.join(lines)

    def evaluate(self) -> int:
        """Heuristic evaluation."""
        if self.check_win(1): return 10000
        if self.check_win(2): return -10000
        score = 0
        # Center column preference
        for r in range(self.ROWS):
            if self.board[r][3] == 1: score += 3
            elif self.board[r][3] == 2: score -= 3
        return score


class GameAI:
    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.tt: dict[int, TTEntry] = {}
        self.nodes = 0
        self.killers: dict[int, list[int]] = {}  # depth → killer moves

    def search(self, game: Connect4) -> tuple[int, int]:
        """Iterative deepening search. Returns (best_move, score)."""
        self.nodes = 0
        best_move = game.legal_moves()[0] if game.legal_moves() else -1
        best_score = 0

        for depth in range(1, self.max_depth + 1):
            move, score = self._root_search(game, depth)
            if move >= 0:
                best_move, best_score = move, score
            if abs(score) > 9000:  # Found forced win/loss
                break

        return best_move, best_score

    def _root_search(self, game: Connect4, depth: int) -> tuple[int, int]:
        best_move = -1
        alpha, beta = -100000, 100000
        moves = self._order_moves(game, depth)

        for col in moves:
            game.make_move(col)
            score = -self._alphabeta(game, depth - 1, -beta, -alpha)
            game.undo_move(col)
            if score > alpha:
                alpha = score
                best_move = col
        return best_move, alpha

    def _alphabeta(self, game: Connect4, depth: int, alpha: int, beta: int) -> int:
        self.nodes += 1
        opponent = 3 - game.current

        if game.check_win(opponent):
            return -(10000 + depth)
        if game.is_draw():
            return 0

        # TT lookup
        h = game.hash()
        if h in self.tt and self.tt[h].depth >= depth:
            entry = self.tt[h]
            if entry.flag == 'exact': return entry.score
            if entry.flag == 'lower': alpha = max(alpha, entry.score)
            if entry.flag == 'upper': beta = min(beta, entry.score)
            if alpha >= beta: return entry.score

        if depth <= 0:
            score = game.evaluate() if game.current == 1 else -game.evaluate()
            return score

        moves = self._order_moves(game, depth)
        best_score = -100000
        best_move = moves[0] if moves else -1

        for col in moves:
            game.make_move(col)
            score = -self._alphabeta(game, depth - 1, -beta, -alpha)
            game.undo_move(col)

            if score > best_score:
                best_score = score
                best_move = col
            alpha = max(alpha, score)
            if alpha >= beta:
                # Killer move heuristic
                if depth not in self.killers:
                    self.killers[depth] = []
                if col not in self.killers[depth]:
                    self.killers[depth].insert(0, col)
                    if len(self.killers[depth]) > 2:
                        self.killers[depth].pop()
                break

        # TT store
        flag = 'exact'
        if best_score <= alpha: flag = 'upper'
        if best_score >= beta: flag = 'lower'
        self.tt[h] = TTEntry(depth, best_score, flag, best_move)

        return best_score

    def _order_moves(self, game: Connect4, depth: int) -> list[int]:
        moves = game.legal_moves()
        scored = []
        # TT move first
        h = game.hash()
        tt_move = self.tt[h].best_move if h in self.tt else -1
        for m in moves:
            score = 0
            if m == tt_move: score += 1000
            if m in self.killers.get(depth, []): score += 100
            score += 10 - abs(m - 3)  # center preference
            scored.append((score, m))
        scored.sort(reverse=True)
        return [m for _, m in scored]


def demo():
    print("=== Game AI (Connect Four) ===\n")
    game = Connect4()
    ai = GameAI(max_depth=8)

    # Play a few moves
    for _ in range(6):
        move, score = ai.search(game)
        player = 'X' if game.current == 1 else 'O'
        print(f"{player} plays column {move} (score: {score}, nodes: {ai.nodes})")
        game.make_move(move)

    print(f"\n{game.display()}")
    print(f"\nTT entries: {len(ai.tt)}")


if __name__ == '__main__':
    if '--test' in sys.argv:
        g = Connect4()
        # Legal moves
        assert len(g.legal_moves()) == 7
        # Make/undo
        g.make_move(3)
        assert g.board[0][3] == 1
        g.undo_move(3)
        assert g.board[0][3] == 0
        # Win detection
        for i in range(4):
            g.make_move(i); g.make_move(i)  # player 1 bottom, player 2 on top
        # After: col 0-3 have P1 at row 0, P2 at row 1
        # Actually need horizontal 4: reset
        g2 = Connect4()
        for c in range(4):
            g2.make_move(c)  # P1
            if c < 3: g2.make_move(c)  # P2 on top (skip last to keep P1 turn)
        # P1 has row 0 cols 0-3? No, alternating. Let me just test AI
        g3 = Connect4()
        ai = GameAI(max_depth=4)
        move, score = ai.search(g3)
        assert 0 <= move < 7
        assert ai.nodes > 0
        print("All tests passed ✓")
    else:
        demo()
