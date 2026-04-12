# FINAL_REPORT

## 1. Task Setup

This project studies cooperative behavior in a custom Google Research Football scenario:

* scenario: `two_v_two_plus_goalkeepers`
* left team: 2 controlled players + 1 goalkeeper
* right team: 2 built-in opponents + 1 goalkeeper

The comparison is organized along three dimensions:

* **Architecture**
  * `shared_ppo`: one shared actor policy for both controlled players
  * `mappo_id_cc`: shared actor with agent identity input and a centralized critic

* **Reward**
  * `R2`: progress-based reward, mainly encouraging forward ball progression
  * `R3`: assist-oriented reward, designed to favor cooperation and passing-related behavior

* **Multi-agent mode**
  * parameter sharing means both agents use the same policy parameters
  * MAPPO-style centralized critic uses global team information during training to improve coordination

---

## 2. Brief Introduction: PPO and MAPPO

**PPO** (Proximal Policy Optimization) is a policy-gradient reinforcement learning method that updates the policy conservatively, which makes training relatively stable and simple. In the shared-parameter setting, both controlled players use the same policy, so symmetric behavior is a natural outcome unless the environment or observations strongly force role differentiation.

**MAPPO** is the multi-agent extension of PPO used here in a centralized-critic form. The actor still outputs actions for each agent, but the critic is trained with broader team-level information. This does not automatically create cooperation, but it usually gives better learning signals for coordinated behavior than fully independent local evaluation.

---

## 3. Result Summary from Representative Videos

In the final qualitative comparison, one representative video is selected for each main setting:

* `R2 + shared_ppo`
* `R3 + shared_ppo`
* `R3 + mappo_id_cc`

The clearest behavior pattern is:

* under **shared parameters**, the two controlled players often behave very similarly
* under **MAPPO**, the two controlled players are more likely to behave differently and take more distinct roles

This pattern is meaningful when linked back to the design choices above.

First, with **shared PPO**, both players are controlled by the same policy and receive highly similar learning updates. In a symmetric 2v2 environment, this naturally encourages mirrored or near-identical behavior. Even when `R3` improves cooperation compared with `R2`, the shared-parameter setting still tends to produce similar movement logic for both agents.

Second, with **MAPPO**, the centralized critic provides a stronger team-level training signal. This makes it easier for the learning process to assign different functional behavior to the two players, even if the actor side is still closely related. As a result, the two controlled agents appear less like duplicates and more like coordinated teammates with different responses to the same situation.

Finally, reward and architecture play different roles:

* `R3` is what pushes the system toward cooperation-related behavior at all
* `MAPPO` is what makes that cooperation look less symmetric and more role-differentiated

So the main conclusion is not only that cooperation improves with reward shaping, but also that **centralized multi-agent training helps transform cooperation from similar parallel behavior into more distinct team behavior**.
