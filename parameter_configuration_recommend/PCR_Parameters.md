# The Important Parameters of PCR Model (TD3)

This document lists the key parameters used to **train** and **fine‑tune** the **PCR model** built on **TD3**. It includes clear defaults and small, practical ranges for exploration.

---

## The description of the parameters and their default values

| Scope | Parameter |    Default | Description                                                                        |
|---|---|-----------:|------------------------------------------------------------------------------------|
| **Env** | `max_steps` |   **5000** | Max number of recommendation steps per episode.                                    |
|  | `nochange_steps` |    **200** | Max number of consecutive steps within an episode without performance improvement. |
|  | `pec_reward` |      **1** | Positive environment reward scalar.                                                |
|  | `memory-size` | **100000** | Number of stored experiences.                                                      |
| **TD3** | `clr` |   **1e-4** | Learning rate of the critic network.                                               |
|  | `alr` |   **1e-5** | Learning rate of the actor network.                                                |
|  | `tau` |   **1e-4** | Soft target‑network update coefficient.                                            |
|  | `sigma` |    **0.2** | Exploration noise standard deviation.                                              |
|  | `delay_time` |      **2** | Update frequency of the actor network.                                             |
|  | `batch_size` |    **128** | Minibatch size for updates.                                                        |
| **Training / Online** | `epoches` |   **8000** | Max number  of **training** episodes.                                              |
|  | `test_epoches` |    **250** | Max number  of **online‑tuning** episodes.                                         |
|  | `nochange_episodes` |    **150** | Max number of consecutive episodes whose step count equals `nochange_steps`.       |

> These defaults have been validated to yield excellent training and tuning performance in practice.

---

## Small, Safe Tweaks

- **Episode length vs. wall‑time**
  - Lower `args_r.max_steps` *below 5000* if you need shorter episodes and faster iterations.
- **Plateau tolerance inside an episode**
  - Use `args_r.nochange_steps ∈ {200, 400}`.  
    - If you choose **200**, set `args_r.epoches = 8000` (or smaller if needed).  
    - If you choose **400**, set `args_r.epoches = 4500` (or smaller if needed).
- **Batching**
  - Try `args_r.batch_size ∈ {128, 256}`.
- **Plateau tolerance across episodes**
  - Try `args_r.nochange_episodes ∈ {100, 150}`.
- **Everything else**
  - The remaining parameters can be kept at their **default values**.

