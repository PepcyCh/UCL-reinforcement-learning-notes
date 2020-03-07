#include <bits/stdc++.h>

#define LOG(expr) std::cout << expr << std::endl;

template <typename T>
class HashFunc {
  public:
    size_t operator()(const T &obj) const {
        return obj.Hash();
    }
};

double Lerp(double a, double b, double t) {
    return a + t * (b - a);
}

/*
 * Sarsa
 *
 * Routine
 *   Initialize action-value function
 *   Do updates for each episode
 *     Get initial state S
 *     Choose action A
 *     For each step
 *       Get R, S' from S, A
 *       Choose action A'
 *       Update Q(S, A)
 *       S, A <- S', A'
 *
 * State should implement
 *   Hash() (or std::hash())
 *   CanDoAction(action)
 *   NextAction(values, epsilon)
 *   NextState(action)
 *   CalcReward(action)
 *   static GetAllStates()
 *   static GetAllActions()
 *   static GetInitialState()
 *
 * Action should implement
 *   Hash() (or std::hash())
 */

template <typename TState, typename TAction,
          typename StateHash = std::hash<TState>,
          typename ActionHash = std::hash<TAction>>
class Sarsa {
  public:
    Sarsa(double gamma = 0.9, double alpha = 0.5) : gamma(gamma), alpha(alpha) {
        all_states = State::GetAllStates();
        all_actions = State::GetAllActions();
    }

    void Train(int n_episodes) {
        auto total_start = std::chrono::high_resolution_clock::now();

        Initialize();
        LOG("Initialization finished");
        for (int i = 0; i < n_episodes; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            double epsilon = 1.0 / (i + 1.0);
            State state = State::GetInitialState();
            Action action = state.NextAction(values[state], epsilon);
            while (!state.IsTerminal()) {
                double reward = state.CalcReward(action);
                State state_p = state.NextState(action);
                Action action_p = state_p.NextAction(values[state_p], epsilon);
                values[state][action] = Lerp(values[state][action],
                        reward + gamma * values[state_p][action_p], alpha);
                state = state_p;
                action = action_p;
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time = end - start;
            LOG("Episode " << i + 1 << ", time used: " << time.count() << "s");
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = total_end - total_start;
        LOG("Total time used: " << total_time.count() << "s");
    }

  protected:
    using State = TState;
    using States = std::unordered_set<State, StateHash>;
    using Action = TAction;
    using Actions = std::unordered_set<Action, ActionHash>;
    using ActionValues = std::unordered_map<State,
          std::unordered_map<Action, double, ActionHash>, StateHash>;

    void Initialize() {
        for (const auto &state : all_states) {
            if (state.IsTerminal()) {
                for (const auto &action : all_actions) {
                    values[state][action] = 0.0;
                }
            } else {
                for (const auto &action : all_actions) {
                    if (state.CanDoAction(action)) {
                        values[state][action] = 0.0;
                    } else {
                        values[state][action] = -1.0 / 0.0;
                    }
                }
            }
        }
    }

    double gamma, alpha;
    States all_states;
    Actions all_actions;
    ActionValues values;
};

const int kLenHorizontal = 12;
const int kLenVertical = 4;

std::uniform_real_distribution rnd_uniform(0.0, 1.0);
std::random_device rnd_rd;
std::mt19937 rnd_gen(rnd_rd());

class CliffWalkingState {
  public:
    CliffWalkingState(int x, int y) : x(x), y(y) {}

    using Action = int;
    using Actions = std::unordered_set<Action>;
    using States = std::unordered_set<CliffWalkingState,
          HashFunc<CliffWalkingState>>;
    using Values = std::unordered_map<Action, double>;

    static States GetAllStates() {
        static States all_states;
        static bool first = true;
        if (first) {
            for (int i = 0; i < kLenVertical; i++) {
                for (int j = 0; j < kLenHorizontal; j++) {
                    all_states.emplace(i, j);
                }
            }
        }
        return all_states;
    }

    static Actions GetAllActions() {
        static Actions all_actions;
        static bool first = true;
        if (first) {
            all_actions.insert(0);
            all_actions.insert(1);
            all_actions.insert(2);
            all_actions.insert(3);
        }
        return all_actions;
    }

    static CliffWalkingState GetInitialState() {
        return CliffWalkingState(0, 0);
    }

    bool CanDoAction(Action action) const {
        CliffWalkingState state_p = NextState(action);
        return state_p.x >= 0 && state_p.x < kLenVertical && state_p.y >= 0 &&
            state_p.y < kLenHorizontal;
    }

    bool IsTerminal() const {
        return x == 0 && y == kLenHorizontal - 1;
    }

    Action NextAction(const Values &values, double epsilon) const {
        double rnd = rnd_uniform(rnd_gen);
        std::vector<Action> actions;
        Actions all_actions = GetAllActions();
        if (rnd < epsilon) { // random
            for (auto action : all_actions) {
                if (CanDoAction(action)) {
                    actions.push_back(action);
                }
            }
        } else { // greedy (random from maxima)
            double max = std::numeric_limits<double>::lowest();
            for (auto action : all_actions) {
                double value = values.at(action);
                if (CanDoAction(action) && value > max) {
                    max = value;
                }
            }
            for (auto action : all_actions) {
                double value = values.at(action);
                if (CanDoAction(action) && std::abs(value - max) < 1e-6) {
                    actions.push_back(action);
                }
            }
        }
        int ind = rnd_gen() % actions.size();
        assert(actions.size() > 0);
        return actions[ind];
    }

    CliffWalkingState NextState(Action action) const {
        const static int d[4][2] = {
            {1, 0}, {0, 1}, {-1, 0}, {0, -1}
        };
        int xx = x + d[action][0];
        int yy = y + d[action][1];
        if (xx == 0 && yy > 0 && yy < kLenHorizontal - 1) {
            xx = 0;
            yy = 0;
        }
        return CliffWalkingState(xx, yy);
    }

    double CalcReward(Action action) const {
        const static int d[4][2] = {
            {1, 0}, {0, 1}, {-1, 0}, {0, -1}
        };
        int xx = x + d[action][0];
        int yy = y + d[action][1];
        return xx == 0 && yy > 0 && yy < kLenHorizontal - 1 ? -100 : -1;
    }

    size_t Hash() const {
        return x * kLenHorizontal + y;
    }

    bool operator==(const CliffWalkingState &rhs) const {
        return x == rhs.x && y == rhs.y;
    }

    int x, y;
};

class CliffWalking : public Sarsa<CliffWalkingState, CliffWalkingState::Action,
        HashFunc<CliffWalkingState>> {
  public:
    CliffWalking(double gamma = 0.9, double alpha = 0.5) : Sarsa(gamma, alpha) {}

    void Display() const {
        for (int i = 0; i < kLenVertical; i++) {
            for (int j = 0; j < kLenHorizontal; j++) {
                if (i == 0 && j > 0 && j < kLenHorizontal - 1) {
                    std::cout << "x ";
                } else {
                    State state(i, j);
                    double max = std::numeric_limits<double>::lowest();
                    Action optimal_action = 0;
                    for (auto action : all_actions) {
                        double value = values.at(state).at(action);
                        if (state.CanDoAction(action) && value > max) {
                            max = value;
                            optimal_action = action;
                        }
                    }
                    std::cout << GetActionChar(optimal_action) << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    static char GetActionChar(Action action) {
        return "V>^<"[action];
    }
};

int main() {
    CliffWalking solver;
    solver.Train(100000);
    solver.Display();
    
    return 0;
}
