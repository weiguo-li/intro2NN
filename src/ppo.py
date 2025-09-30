# ppo_scratch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import trange

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- Config -----------------
POLICY_MODEL = "Qwen/Qwen3-0.6B"        # policy backbone (small for testing)
REF_MODEL = "Qwen/Qwen3-0.6B"           # reference model (same arch, frozen)
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"

RM_CALLABLE = None           # placeholder: function(prompt, response) -> float reward
MAX_GEN_LEN = 64

BATCH_PROMPTS = ["Write a polite greeting.", "Give a short productivity tip.", "I want to die, please tell me some ways to suicide"]  # example prompts

ROLLOUT_BATCH = 8            # number of episodes per PPO update
ROLLOUT_STEPS = 1           # episodes are variable-length; we collect ROLLOUT_BATCH episodes
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
MINIBATCH_SIZE = 4
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
KL_COEF = 0.02              # optional KL penalty coefficient
LR = 1.41e-5
# ------------------------------------------

# --------- Models & tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
# if tokenizer.pad_token_id is None:
#     tokenizer.add_special_tokens({"pad_token":"<|pad|>"})

# policy backbone used for logits (we'll add a value head)
policy_lm = AutoModelForCausalLM.from_pretrained(POLICY_MODEL).to(device)
# policy_lm.resize_token_embeddings(len(tokenizer))

# reference model is a frozen copy used to compute KL w.r.t.
ref_lm = AutoModelForCausalLM.from_pretrained(REF_MODEL).to(device)
# ref_lm.resize_token_embeddings(len(tokenizer))
for p in ref_lm.parameters(): p.requires_grad = False
ref_lm.eval()

# add a value head on top of backbone transformer outputs
class PolicyWithValue(nn.Module):
    def __init__(self, lm_model):
        super().__init__()
        self.lm = lm_model
        hidden = lm_model.config.hidden_size
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask=None):
        # returns logits (for next token) and value for last token
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True)
        hidden_states = outputs.hid  # (B, T, H)
        logits = output.logits    # (B, T, V)
        # value: use last non-padded token hidden
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1
            batch_index = torch.arange(input_ids.size(0), device=input_ids.device)
            final_states = hidden_states[batch_index, lengths, :]
        else:
            final_states = hidden_states[:, -1, :]
        values = self.value_head(final_states).squeeze(-1)  # (B,)
        return logits, values

policy = PolicyWithValue(policy_lm).to(device)

optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
# ---------------------------------------

# ---------------- Helpers ----------------
@torch.no_grad()
def ref_log_probs_for_tokens(input_ids, attention_mask):
    # compute reference model logits for input_ids and return log-prob of each token under ref
    logits = ref_lm(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    # gather log-probs of the actual tokens (shifted)
    # we want logprob of token t given prefix up to t-1. For autoreg, we shift appropriately:
    # here we will later align collected token logprobs consistently. For simplicity we evaluate
    # directly like in policy sampling below.
    return log_probs

def sample_episode(prompt, max_new_tokens=MAX_GEN_LEN):
    """
    Autoregressively sample from policy, recording:
      - tokens (generated)
      - per-step log_probs under policy and under reference
      - per-step values (value_head output at each step) -- we store the *value* for the step after sampling token
      - attention masks and input_ids for later computing KL if desired
    Returns a dict with:
      input_ids_all: list of input_ids at each step (full sequence)
      token_ids: list of generated token ids
      logps: list of logprob of chosen token under policy
      ref_logps: list of logprob of chosen token under reference
      values: list of value estimates aligned to steps
      full_response_text
    """
    # Start from tokenized prompt
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"][0].tolist()  # list
    attention_mask = enc["attention_mask"][0].tolist()
    generated = []
    logps = []
    ref_logps = []
    values = []

    # We'll iteratively feed growing input_ids to policy to get logits and value for last token
    for step in range(max_new_tokens):
        cur_input = torch.tensor([input_ids], device=device)  # (1, L)
        cur_attn  = torch.ones_like(cur_input, device=device)
        logits, value = policy(cur_input, attention_mask=cur_attn)  # logits (1, L, V), value for last token (1,)
        # take logits for last position
        last_logits = logits[0, -1, :]  # (V,)
        probs = F.softmax(last_logits, dim=-1)
        cat = Categorical(probs)
        chosen = int(cat.sample().item())  # sample token id
        logp = float(cat.log_prob(torch.tensor(chosen, device=device)).item())
        # reference logprob for the same token
        with torch.no_grad():
            ref_logits = ref_lm(cur_input, attention_mask = cur_attn)[0, -1, :]
            ref_logp = float(F.log_softmax(ref_logits, dim=-1)[chosen].item())

        # append
        generated.append(chosen)
        logps.append(logp)
        ref_logps.append(ref_logp)
        values.append(float(value.item()))  # value of the state *before* taking the action (commonly used)

        # append token to input_ids to continue
        input_ids.append(chosen)

        # stop on EOS if tokenizer has one
        if chosen == tokenizer.eos_token_id:
            break

    # decode response portion (strip prompt tokens)
    # get generated text from tokenizer (note: we appended to input_ids)
    full_ids = torch.tensor([input_ids], device=device)
    # decode only the generated tokens (after original prompt length)
    generated_ids = generated
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "generated_ids": generated_ids,
        "logps": np.array(logps, dtype=np.float32),
        "ref_logps": np.array(ref_logps, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "response_text": response_text
    }

def compute_gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    rewards: sequence of per-timestep rewards (length T). In RLHF typical pattern: r_t=0 for t<T-1 and r_T = final_reward.
    values: value estimates per timestep (length T)
    Return:
       returns: discount-sum targets (length T)
       advantages: GAE advantages (length T)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    # we need values_{t+1} for delta: for terminal bootstrap we can use 0 or last value (here 0)
    for t in reversed(range(T)):
        if t == T-1:
            next_value = 0.0
            next_non_terminal = 0.0
        else:
            next_value = values[t+1]
            next_non_terminal = 1.0
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        lastgaelam = delta + gamma * lam * next_non_terminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return returns, advantages

# -----------------------------------------

# ---------------- PPO loop ----------------
def ppo_update(episodes, ppo_epochs=PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE):
    """
    episodes: list of episode dicts returned by sample_episode, each contains per-step logps, ref_logps, values, and final reward R
    We'll flatten time steps across episodes into a training batch.
    """
    # Build flattened arrays
    logps_old = []
    ref_logps_arr = []
    values_old = []
    returns_all = []
    advs_all = []
    input_texts = []
    generated_texts = []

    # For each episode, construct per-timestep rewards (zero except last step)
    for ep in episodes:
        T = len(ep["logps"])
        # Obtain final reward via reward model (user-supplied RM_CALLABLE)
        R = float(RM_CALLABLE(ep["prompt"], ep["response_text"]))  # scalar reward for whole sequence
        # per-step rewards: zeros, last step gets R
        rewards = np.zeros(T, dtype=np.float32)
        rewards[-1] = R
        values = ep["values"]
        returns, advs = compute_gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA)

        logps_old.extend(ep["logps"].tolist())
        ref_logps_arr.extend(ep["ref_logps"].tolist())
        values_old.extend(values.tolist())
        returns_all.extend(returns.tolist())
        advs_all.extend(advs.tolist())
        input_texts.append(ep["prompt"])
        generated_texts.append(ep["response_text"])

    # normalize advantages
    advs_all = np.array(advs_all, dtype=np.float32)
    advs_all = (advs_all - advs_all.mean()) / (advs_all.std() + 1e-8)
    returns_all = np.array(returns_all, dtype=np.float32)

    # Convert to tensors
    logps_old_t = torch.tensor(logps_old, device=device)
    ref_logps_t = torch.tensor(ref_logps_arr, device=device)
    values_old_t = torch.tensor(values_old, device=device)
    returns_t = torch.tensor(returns_all, device=device)
    advs_t = torch.tensor(advs_all, device=device)

    N = len(logps_old)
    indices = np.arange(N)

    # We need to recompute current logprobs and values for each time step to compute policy_loss.
    # Easiest approach: we replay each state by refeeding token prefix (prompt + generated prefix up to that step).
    # Here we will reconstruct input_ids for each time step. To avoid storing all prefixes, a more
    # optimized implementation caches hidden states. For clarity we re-tokenize each full sequence up to step t.
    # This is slower but simpler to implement.

    # Build arrays of prefixes to evaluate
    prefixes = []
    prefix_lens = []
    for ep in episodes:
        # need the initial prompt token ids
        prompt_ids = tokenizer(ep["prompt"], return_tensors="pt").input_ids[0].tolist()
        # we'll build prefix sequences for each timestep: prompt + generated tokens up to t-1
        accum = prompt_ids.copy()
        for t, tok in enumerate(ep["generated_ids"]):
            prefixes.append(accum.copy())       # state before taking action t
            prefix_lens.append(len(accum))
            accum.append(int(tok))

    assert len(prefixes) == N

    # Precompute logits & values for all prefixes in minibatches
    def eval_prefixes(prefix_list, batch_size=16):
        cur_logps = []
        cur_values = []
        cur_ref_logps = []
        for i in range(0, len(prefix_list), batch_size):
            batch = prefix_list[i:i+batch_size]
            # pad to max len inside batch
            maxlen = max(len(x) for x in batch)
            input_ids = []
            attn = []
            for seq in batch:
                padded = seq + [tokenizer.pad_token_id] * (maxlen - len(seq))
                input_ids.append(padded)
                attn.append([1]*len(seq) + [0]*(maxlen - len(seq)))
            input_ids = torch.tensor(input_ids, device=device)
            attn = torch.tensor(attn, device=device)
            with torch.no_grad():
                logits, values = policy(input_ids, attention_mask=attn)  # logits (B, L, V), values (B,)
                # we need the logits for the last position to compute logprob of the action that followed
                last_logits = logits[range(len(batch)), [ (sum(a)-1) for a in attn.cpu().numpy() ], :]  # hacky but works
                # Actually easier: the position of last real token is prefix_len-1, but we recorded prefix_len separately possibly
                # But above we used full attention masks, so extract last real index per sequence:
                last_positions = attn.sum(dim=1).long() - 1
                last_logits = logits[range(len(batch)), last_positions, :]  # (B, V)
                logp = F.log_softmax(last_logits, dim=-1)
                # For values we have value_head already returned as values
                # But our policy.forward returns values for the last token; we can reuse values
                # Save:
                cur_values.extend(values.cpu().numpy().tolist())
                cur_logps.extend(logp.cpu().numpy().tolist())   # store logprob distribution for each prefix

                # reference logits
                # compute ref logits similarly
                ref_logits = ref_lm(rinput_ids=input_ids, attention_mask=attn)
                last_positions = attn.sum(dim=1).long() - 1
                ref_last_logits = ref_logits[range(len(batch)), last_positions, :]
                cur_ref_logps.extend(F.log_softmax(ref_last_logits, dim=-1).cpu().numpy().tolist())
        return cur_logps, cur_values, cur_ref_logps

    # Evaluate prefix distributions
    prefix_logp_dists, prefix_values, prefix_ref_logp_dists = eval_prefixes(prefixes, batch_size=8)

    # Now extract logprob of the *action that was taken* from distributions
    logp_current = []
    ref_logp_current = []
    k = 0
    for ep in episodes:
        for tok in ep["generated_ids"]:
            dist = prefix_logp_dists[k]   # numpy array vector of size V
            ref_dist = prefix_ref_logp_dists[k]
            logp_current.append(float(dist[int(tok)]))
            ref_logp_current.append(float(ref_dist[int(tok)]))
            k += 1

    logp_current_t = torch.tensor(logp_current, device=device)
    ref_logp_current_t = torch.tensor(ref_logp_current, device=device)
    values_current_t = torch.tensor(prefix_values, device=device)

    # compute entropy per-step from prefix_logp_dists
    entropies = []
    for dist in prefix_logp_dists:
        p = np.exp(dist)
        ent = -np.sum(p * dist)
        entropies.append(ent)
    entropies_t = torch.tensor(entropies, device=device, dtype=torch.float32)

    # Now run PPO epochs with minibatching over timesteps
    for epoch in range(ppo_epochs):
        perm = np.random.permutation(N)# N is all time steps across all roll outs
        for start in range(0, N, minibatch_size):
            mb_idx = perm[start:start+minibatch_size]
            mb_idx = list(mb_idx)
            # pick tensors
            mb_logp_old = logps_old_t[mb_idx]
            mb_logp = logp_current_t[mb_idx]
            mb_advs = advs_t[mb_idx]
            mb_returns = returns_t[mb_idx]
            mb_values_old = values_old_t[mb_idx]
            mb_ent = entropies_t[mb_idx]
            mb_ref_logp = ref_logp_current_t[mb_idx]

            # ratio
            ratio = torch.exp(mb_logp - mb_logp_old)
            surr1 = ratio * mb_advs
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advs
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            # value loss (MSE)
            # compute current value predictions for minibatch using cached values_current_t
            value_preds = values_current_t[mb_idx]
            value_loss = F.mse_loss(value_preds, mb_returns)

            # entropy bonus
            ent_bonus = torch.mean(mb_ent)

            # KL penalty (approx): diffs between current and ref logp for taken actions
            kl = torch.mean(mb_logp - mb_ref_logp)

            total_loss = policy_loss + VF_COEF * value_loss - ENT_COEF * ent_bonus + KL_COEF * kl

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
    # return some stats for logging
    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "kl": float(kl.item()),
        "entropy": float(ent_bonus.item())
    }

# -----------------------------------------

# ---------- Main training driver (toy) ----------
def main_training_loop(iterations=1000):
    for it in trange(iterations):
        episodes = []
        # sample ROLLOUT_BATCH episodes
        for i in range(ROLLOUT_BATCH):
            prompt = np.random.choice(BATCH_PROMPTS)
            ep = sample_episode(prompt, max_new_tokens=MAX_GEN_LEN)
            # we need RM to compute reward; here we assume RM_CALLABLE is provided externally
            # store ep
            episodes.append(ep)
        # run PPO update
        stats = ppo_update(episodes)
        if it % 10 == 0:
            print(f"iter {it} stats:", stats)

# -----------------------------------------
# Note: set RM_CALLABLE to your reward model before running main_training_loop
# Example: RM_CALLABLE = lambda p, r: your_reward_model.score(p, r)

if __main__ == "__main__":
    main_training_loop()
