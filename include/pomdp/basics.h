#ifndef POMDP_BASICS_H
#define POMDP_BASICS_H

class State {
}

class Action {
}

class Observation {
}

class History {
}

class Belief {
}

class ObservationModel {
public:
    double probability(const Observation &observation,
                       const State &next_state,
                       const Action &action);

    State sample(const State &next_state,
                 const Action &action);

    State argmax(const State &next_state,
                 const Action &action);
}

class TransitionModel {
public:
    double probability(const State &next_state,
                       const State &next_state,
                       const Action &action);

    State sample(const State &state,
                 const Action &action);

    State argmax(const State &state,
                 const Action &action);
}

class RewardModel {
public:
    double probability(const State &next_state,
                       const State &next_state,
                       const Action &action);

    double sample(const State &state,
                  const Action &action,
                  const State &next_state);

    double argmax(const State &state,
                  const Action &action,
                  const State &next_state);
}

class PolicyModel {
public:
    Action sample(State &state);
    void getAllActions(State &state, History &history);
    void getAllActions(State &state);
}

class Agent {
public:
    Agent(Belief init_belief, PolicyModel pi, TransitionModel T,
          ObservationModel O, RewardModel R)
        : belief_(init_belief), pi_(pi), T_(T), O_(O), R_(R) {}

private:
    Belief belief_;
    TransitionModel T_;
    ObservationModel O_;
    RewardModel R_;
    PolicyModel pi_;
}

class Environment {
public:
    Environment(State &init_state, TransitionModel T, RewardModel R)
        : state_(init_state), T_(T), R_(R) {}
private:
    State state_;
    TransitionModel T_;
    RewardModel R_;
}

#endif
