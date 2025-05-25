import numpy as np
from utils import result_function

class PokerEnvironment:
    def __init__(self, config):
        self.M = config['M']
        self.deck = np.arange(52)
        self.max_bet = 3 * self.M
        self.state_dim = 9  # [P1 카드1, P1 카드2, P2 카드1, P2 카드2, 커뮤니티 카드1-5]
        self.reset()

    def reset(self):
        np.random.shuffle(self.deck)
        self.player1_hand = self.deck[:2]
        self.player2_hand = self.deck[2:4]
        self.community_cards = self.deck[4:9]
        self.pot = 2 * self.M
        self.player1_bet = self.M
        self.player2_bet = self.M
        self.current_step = 0
        self.betting_history = []
        self.player1_money = 1000
        self.player2_money = 1000
        return self.get_full_state()

    def get_full_state(self):
        return np.concatenate([self.player1_hand, self.player2_hand, self.community_cards])

    def get_actor_state(self, player, nn_index):
        if player == 'A':
            own_hand = self.player1_hand
            opponent_bets = [b[1] for b in self.betting_history if b[0] == 'B']
        else:
            own_hand = self.player2_hand
            opponent_bets = [b[1] for b in self.betting_history if b[0] == 'A']

        if nn_index == 0:  # nn1
            return np.concatenate([own_hand, [self.pot]])
        elif nn_index == 1:  # nn2
            return np.concatenate([own_hand, self.community_cards[:3], opponent_bets[:1], [self.pot]])
        elif nn_index == 2:  # nn3
            return np.concatenate([own_hand, self.community_cards[:4], opponent_bets[:2], [self.pot]])
        elif nn_index == 3:  # nn4
            return np.concatenate([own_hand, self.community_cards, opponent_bets[:3], [self.pot]])
        elif nn_index == 4:  # nn5
            return np.concatenate([own_hand, opponent_bets[:1], [self.pot]])
        elif nn_index == 5:  # nn6
            return np.concatenate([own_hand, self.community_cards[:3], opponent_bets[:2], [self.pot]])
        elif nn_index == 6:  # nn7
            return np.concatenate([own_hand, self.community_cards[:4], opponent_bets[:3], [self.pot]])
        elif nn_index == 7:  # nn8
            return np.concatenate([own_hand, self.community_cards, opponent_bets[:4], [self.pot]])

    def step(self, action1, action2):
        turn_order = [('A', 'B'), ('B', 'A'), ('A', 'B'), ('B', 'A')]
        first, second = turn_order[self.current_step]
        players = {'A': (action1, self.player1_money, self.player1_bet), 'B': (action2, self.player2_money, self.player2_bet)}
        C = abs(self.player1_bet - self.player2_bet)
        U = self.pot + 2 * C if C > 0 else 3 * self.M

        for player, (x, money, bet) in players.items():
            if player == first:
                Y, action_type = self._process_first_action(x, C, money, U)
            else:
                Y, action_type = self._process_second_action(x, C, money, U)
            
            if player == 'A':
                self.player1_bet += Y
                self.player1_money -= Y
            else:
                self.player2_bet += Y
                self.player2_money -= Y
            self.pot += Y
            self.betting_history.append((player, Y))

        done = False
        reward1, reward2 = 0, 0
        if 'Fold' in [action_type] or self.current_step == 3:
            done = True
            winner = result_function(self.get_full_state())
            if winner == 'A':
                reward1 = self.pot / U
                reward2 = -self.player2_bet / U
            else:
                reward1 = -self.player1_bet / U
                reward2 = self.pot / U
        self.current_step += 1
        return self.get_full_state(), reward1, reward2, done

    def _process_first_action(self, x, C, money, U):
        if x >= C + self.M:
            Y = C + int((x - C) / self.M) * self.M
            Y = min(Y, money, U)
            return Y, 'Raise'
        elif 0 < x < C:
            return 0, 'Fold'
        else:
            return C, 'Call'

    def _process_second_action(self, x, C, money, U):
        if x >= 2 * self.M:
            Y = self.M + int((x - self.M) / self.M) * self.M
            Y = min(Y, money, U)
            return Y, 'Raise'
        elif 0 < x < self.M:
            return 0, 'Fold'
        else:
            return 0, 'Check'
