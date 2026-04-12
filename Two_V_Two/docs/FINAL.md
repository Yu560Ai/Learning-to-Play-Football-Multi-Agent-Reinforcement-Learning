# Final Results Summary (Two_V_Two Custom Scenario)

## Setup

We evaluate learned policies in the custom environment:

* **Scenario**: `two_v_two_plus_goalkeepers`
* **Left team**: 2 controlled agents + goalkeeper
* **Right team**: 2 built-in opponents + goalkeeper

We compare:

1. **Reward design**

   * `R2`: progress-based reward (baseline)
   * `R3`: cooperation-favorable reward (assist-oriented)

2. **Multi-agent structure**

   * `shared_ppo`: shared parameters
   * `mappo_id_cc`: centralized critic (MAPPO-style)

---

## Evaluation Method

We use **qualitative video analysis** at three training stages:

* Early: `update_000100`
* Mid: `update_000250`
* Late: `update_000490`

Each checkpoint is rendered as a rollout video.

---

## 1. Reward Comparison: R2 vs R3 (shared PPO)

### Videos

* R2/shared:

  * `update_000100`
  * `update_000250`
  * `update_000490`

* R3/shared:

  * `update_000100`
  * `update_000250`
  * `update_000490`

---

### Observations

#### Early stage (~100)

* Both R2 and R3 show mostly **random or weakly structured movement**
* Limited ball control, frequent turnovers
* No visible cooperation yet

#### Mid stage (~250)

* **R2 (progress reward)**:

  * Agents begin moving forward more consistently
  * Behavior remains largely **individualistic**
  * Possession often held by a single agent

* **R3 (cooperation reward)**:

  * Emergence of **passing attempts**
  * More **team-aware positioning**
  * Occasional sequences where one agent supports another

#### Late stage (~490)

* **R2**:

  * Stable forward progression behavior
  * Still mostly **single-agent play**
  * Limited evidence of coordinated actions

* **R3**:

  * Clearer **multi-step interaction patterns**
  * Repeated **pass-like behaviors**
  * More structured attack formation
  * Still imperfect, but visibly more cooperative

---

### Conclusion (Reward)

* `R2` produces a **strong individual baseline**
* `R3` introduces **cooperation structure**
* Cooperation is **emergent but not fully stable**

---

## 2. Structure Comparison: Shared PPO vs MAPPO (R3 reward)

### Videos

* R3/shared:

  * `update_000100`
  * `update_000250`
  * `update_000490`

* R3/mappo_id_cc:

  * `update_000100`
  * `update_000250`
  * `update_000490`

---

### Observations

#### Early stage (~100)

* Both methods behave similarly
* No clear advantage from centralized critic

#### Mid stage (~250)

* **Shared PPO**:

  * Intermittent cooperation signals
  * Passing appears but is inconsistent

* **MAPPO (centralized critic)**:

  * More **coordinated positioning**
  * Slightly more stable interaction patterns
  * Better spatial awareness between agents

#### Late stage (~490)

* **Shared PPO**:

  * Cooperation remains present but unstable
  * Behavior fluctuates between coordinated and individual

* **MAPPO**:

  * More consistent **team structure**
  * Better **synchronization between agents**
  * Still no perfect coordination, but visibly improved organization

---

### Conclusion (Structure)

* Centralized critic (MAPPO) improves:

  * **stability of cooperation**
  * **team-level organization**
* Effect is **incremental**, not dramatic

---

## Overall Conclusion

* Reward design is the **primary driver** of cooperation:

  * Without R3, no meaningful teamwork emerges

* Multi-agent structure provides a **secondary improvement**:

  * MAPPO stabilizes but does not create cooperation alone

* Within limited training budget:

  * Cooperation is **visible but incomplete**
  * Results are sufficient to demonstrate:

    * emergence of team behavior
    * sensitivity to reward and architecture

---

## Key Takeaway

Even in a simple 2v2 setting:

* **Cooperation does not emerge automatically**
* It requires:

  * appropriate reward shaping (R3)
  * and benefits from structured learning (MAPPO)

These results highlight the importance of **reward design + multi-agent structure** in learning cooperative behavior.
