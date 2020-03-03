#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <future>
#include <cassert>

/*
 * general codes for policy iteration
 */

#define LOG(expr) std::cout << expr << std::endl;

template <typename T>
class HashFunc {
  public:
    size_t operator()(const T &obj) const {
        return obj.Hash();
    }
};

/*
 * Policy Iteration (parallel)
 *
 * Routine
 *   Loop util policies are kept unchanged:
 *     Do policy elevuation
 *     Do policy improvement greedily
 *
 * Time complexity
 *   O(n^2 m) for both policy elevuation and policy improvement
 *   n - # states
 *   m - # actions
 *
 * State should implement
 *   Hash() (or std::hash())
 *   CanDoAction()
 * Action should implement
 *   Hash() (or std::hash())
 *
 * Class that inherits this one should implement
 *   CalcPss()
 *   CalcImmReward()
 */
template <typename TState, typename TAction,
          typename StateHash = std::hash<TState>,
          typename ActionHash = std::hash<TAction>>
class PolicyIteration {
  public:
    void Train() {
        int iter = 0;
        auto total_start = std::chrono::high_resolution_clock::now();
        while (true) {
            ++iter;

            auto start = std::chrono::high_resolution_clock::now();
            PolicyEvaluation();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_used = end - start;
            LOG("PE time used: " << time_used.count() << "s");

            start = std::chrono::high_resolution_clock::now();
            int n_policy_changed = PolicyImprovement();
            end = std::chrono::high_resolution_clock::now();
            time_used = end - start;
            LOG("PI time used: " << time_used.count() << "s");

            LOG("After " << iter << " iterations, " << n_policy_changed <<
                    " policies changed");
            if (n_policy_changed == 0)
                break;
        }
        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = total_end - total_start;
        LOG("Total iterations: " << iter);
        LOG("Total time used: " << total_time.count() << "s");
    }

  protected:
    using State = TState;
    using States = std::unordered_set<State, StateHash>;
    using Action = TAction;
    using Actions = std::unordered_set<Action, ActionHash>;
    using Reward = double;
    using Values = std::unordered_map<State, Reward, StateHash>;
    using Policy = std::unordered_map<Action, double, ActionHash>;
    using Policies = std::unordered_map<State, Policy, StateHash>;

    void PolicyEvaluation() {
        while (true) {
            Values new_values;
            Reward diff = 0;
            std::vector<std::future<std::pair<Reward, State>>> handles;
            for (const auto &state : all_states) {
                handles.push_back(std::async(std::launch::async,
                        [this, &state]() { return this->CalcPE(state); }));
            }
            for (auto &future : handles) {
                auto [new_value, state] = future.get();
                diff += std::abs(new_value - values[state]);
                new_values[state] = new_value;
            }
            values = new_values;
            LOG("Difference: " << diff);
            if (diff < eps)
                break;
        }
    }

    int PolicyImprovement() {
        Policies new_policies;
        int diff = 0;
        std::vector<std::future<std::pair<Policy, State>>> handles;
        for (const auto &state : all_states) {
            handles.push_back(std::async(std::launch::async,
                    [this, &state]() { return this->DoPI(state); }));
        }
        for (auto &future : handles) {
            auto [new_policy, state] = future.get();
            diff += 1 - (policies[state] == new_policy);
            new_policies[state] = new_policy;
        }
        policies = new_policies;
        return diff;
    }

    std::pair<Reward, State> CalcPE(const State &state) const {
        const Policy &policy = policies.at(state);

        Reward res = 0;
        for (const auto &[action, prob] : policy) {
            Reward imm_reward = CalcImmReward(state, action);
            Reward dis_reward = CalcDisReward(state, action);
            res += (imm_reward + dis_reward * gamma) * prob;
        }
        return std::make_pair(res, state);
    }

    std::pair<Policy, State> DoPI(const State &state) const {
        Reward max = std::numeric_limits<Reward>::lowest();
        int n_max = 0;
        std::unordered_map<Action, Reward> map;

        for (const auto &action : all_actions) {
            if (!state.CanDoAction(action))
                continue;
            Reward value = CalcDisReward(state, action);
            map[action] = value;
            if (value > max) {
                max = value;
                n_max = 1;
            } else if (value == max) {
                ++n_max;
            }
        }
        assert(n_max > 0);

        Policy new_policy;
        double prob = 1.0 / n_max;
        for (const auto &[action, reward] : map) {
            if (reward == max) {
                new_policy[action] = prob;
            }
        }
        return std::make_pair(new_policy, state);
    }

    virtual double CalcPss(const State &state, const State &state_p,
            const Action &action) const = 0;

    virtual Reward CalcImmReward(const State &state,
            const Action &action) const = 0;

    Reward CalcDisReward(const State &state, const Action &action) const {
        Reward res = 0;
        for (const auto &state_p : all_states) {
            res += values.at(state_p) * CalcPss(state, state_p, action);
        }
        return res;
    }

    double gamma, eps;
    States all_states;
    Actions all_actions;
    Values values;
    Policies policies;
};

/*
 *  Car rental problem
 */

const int kMaxCars = 20;
const int kMaxMove = 5;
const int kRentReward = 10;
const int kMoveCost = -2;
const double kAvgReq1 = 3;
const double kAvgRet1 = 3;
const double kAvgReq2 = 4;
const double kAvgRet2 = 2;

constexpr double Poisson(int n, double lambda) {
    return std::exp(-lambda) * std::pow(lambda, n) / std::tgamma(n + 1);
}

class CarRentalState {
  public:
    CarRentalState(int x, int y) : x(x), y(y) {}

    using Action = int;

    int x, y;

    size_t Hash() const {
        return x * (kMaxCars + 1) + y;
    }
    bool operator==(const CarRentalState &rhs) const {
        return x == rhs.x && y == rhs.y;
    }

    bool CanDoAction(Action action) const {
        return (action > 0 && x >= action) || (action < 0 && y >= -action)
            || action == 0;
    }
};

class CarRental : public PolicyIteration<CarRentalState, CarRentalState::Action,
        HashFunc<CarRentalState>> {
  public:
    CarRental(double gamma = 0.9, double eps = 1e-1) {
        this->gamma = gamma;
        this->eps = eps;

        for (int i = -kMaxMove; i <= kMaxMove; i++) {
            all_actions.insert(i);
        }

        for (int i = 0; i <= kMaxCars; i++) {
            for (int j = 0; j <= kMaxCars; j++) {
                State state(i, j);
                values[state] = 0;

                int nAction = 0;
                Policy policy;
                for (Action action : all_actions) {
                    if (state.CanDoAction(action)) {
                        ++nAction;
                        policy[action] = 1;
                    }
                }
                double inv = 1.0 / nAction;
                for (auto &[action, prob] : policy) {
                    prob *= inv;
                }
                policies[state] = policy;

                all_states.insert(state);
            }
        }
    }

    void Display() const {
        for (int i = 0; i <= kMaxCars; i++) {
            for (int j = 0; j <= kMaxCars; j++) {
                const Policy &policy = policies.at(State(i, j));
                auto it = policy.begin();
                Action act = policy.empty() ? it->first : 11;
                std::cout << std::setw(2) << policy.begin()->first << " ";
                if (j == kMaxCars) std::cout << std::endl;
            }
        }
        std::cout << std::endl;
        for (int i = 0; i <= kMaxCars; i++) {
            for (int j = 0; j <= kMaxCars; j++) {
                std::cout << std::setw(2) << values.at(State(i, j)) << " ";
                if (j == kMaxCars) std::cout << std::endl;
            }
        }
    }

  protected:
    double CalcPss(const State &state, const State &state_p,
            const Action &action) const {
        if (!state.CanDoAction(action))
            return 0;

        int new_x = std::min(state.x - action, kMaxCars);
        int new_y = std::min(state.y + action, kMaxCars);
        int delta_x = state_p.x - new_x;
        int delta_y = state_p.y - new_y;

        double pss = 0;
        for (int req1 = std::max(0, -delta_x); req1 <= new_x; req1++) {
            for (int req2 = std::max(0, -delta_y); req2 <= new_y; req2++) {
                int ret1 = req1 + delta_x;
                int ret2 = req2 + delta_y;
                double prob = Poisson(req1, kAvgReq1) * Poisson(ret1, kAvgRet1)
                            * Poisson(req2, kAvgReq2) * Poisson(ret2, kAvgRet2);
                pss += prob;
            }
        }
        assert(pss >= 0);
        return pss;
    }

    Reward CalcImmReward(const State &state, const Action &action) const {
        if (!state.CanDoAction(action))
            return -1.0 / 0.0;
        Reward move_reward = kMoveCost * std::abs(action);

        Reward rent_reward = 0;
        int new_x = std::min(state.x - action, kMaxCars);
        int new_y = std::min(state.y + action, kMaxCars);
        for (int req1 = 0; req1 <= new_x; req1++) {
            for (int req2 = 0; req2 <= new_y; req2++) {
                double prob_req = Poisson(req1, kAvgReq1) * Poisson(req2, kAvgReq2);
                double prob_ret = 0;
                for (int ret1 = 0; new_x - req1 + ret1 <= kMaxCars; ret1++) {
                    for (int ret2 = 0; new_y - req2 + ret2 <= kMaxCars; ret2++) {
                        double prob = Poisson(ret1, kAvgRet1) *
                                      Poisson(ret2, kAvgRet2);
                        prob_ret += prob;
                    }
                }
                rent_reward += prob_req * prob_ret * (req1 + req2) * kRentReward;
            }
        }

        return move_reward + rent_reward;
    }
};

int main() {
    CarRental solver;
    solver.Train();
    solver.Display();

    return 0;
}
