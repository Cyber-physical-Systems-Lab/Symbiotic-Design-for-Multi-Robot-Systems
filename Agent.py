class Agent:
    def __init__(self, x, y):
        self.position = (x, y)

    def get_state(self):
        return self.position

    def move(self, action_index):
        # Actually update position based on action
        self.position = self.simulate_move(action_index)

    def simulate_move(self, action_index):
        # Simulate movement without updating the position
        action_map = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        dx, dy = action_map[action_index]
        new_x = max(0, min(9, self.position[0] + dx))
        new_y = max(0, min(9, self.position[1] + dy))
        return (new_x, new_y)

    def select_action(self, actor_critic):
        return actor_critic.select_action(self.get_state())
