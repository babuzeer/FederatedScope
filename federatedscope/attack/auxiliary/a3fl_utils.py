import math
from collections import defaultdict

import numpy as np


def parse_attacker_ids(attacker_id):
    if isinstance(attacker_id, int):
        return [] if attacker_id < 0 else [attacker_id]
    if isinstance(attacker_id, list):
        return [int(x) for x in attacker_id]
    raise TypeError(f'Unsupported attacker id type: {type(attacker_id)}')


def get_a3fl_cfg(cfg):
    return cfg.attack.a3fl


def get_a3fl_start_round(cfg):
    a3fl_cfg = get_a3fl_cfg(cfg)
    start_round = int(getattr(a3fl_cfg, 'start_round', -1))
    if start_round >= 0:
        return start_round
    return int(getattr(cfg.attack, 'inject_round', 0))


def is_a3fl_active_round(cfg, round_idx):
    start_round = get_a3fl_start_round(cfg)
    if round_idx < start_round:
        return False

    poison_epochs = int(getattr(get_a3fl_cfg(cfg), 'poison_epochs', 0))
    if poison_epochs <= 0:
        return True
    return round_idx < start_round + poison_epochs


def build_a3fl_random_updates_schedule(cfg, sample_client_num):
    a3fl_cfg = get_a3fl_cfg(cfg)
    attacker_ids = parse_attacker_ids(cfg.attack.attacker_id)
    poison_epochs = int(a3fl_cfg.poison_epochs)
    if len(attacker_ids) == 0 or poison_epochs <= 0:
        return {}

    start_round = get_a3fl_start_round(cfg)

    total_slots = poison_epochs * int(sample_client_num)
    poison_slots = min(
        total_slots,
        max(0, int(math.ceil(a3fl_cfg.sample_poison_ratio * total_slots))))
    if poison_slots == 0:
        return {}

    rng = np.random.RandomState(int(cfg.seed) + 2027)
    selected_slots = rng.choice(total_slots, size=poison_slots, replace=False)
    round2count = defaultdict(int)
    for slot in selected_slots:
        round2count[int(slot) // int(sample_client_num)] += 1

    schedule = {}
    for local_round_idx, count in round2count.items():
        select_num = min(count, len(attacker_ids))
        chosen = rng.choice(attacker_ids, size=select_num, replace=False)
        schedule[int(start_round + local_round_idx)] = [
            int(x) for x in chosen.tolist()
        ]
    return schedule


def get_a3fl_active_attacker_ids(cfg, round_idx, sample_client_num):
    a3fl_cfg = get_a3fl_cfg(cfg)
    if not is_a3fl_active_round(cfg, round_idx):
        return []

    attacker_ids = parse_attacker_ids(cfg.attack.attacker_id)
    if len(attacker_ids) == 0:
        return []

    if a3fl_cfg.sample_method == 'random_updates':
        schedule = build_a3fl_random_updates_schedule(cfg, sample_client_num)
        return schedule.get(int(round_idx), [])

    return attacker_ids


def should_a3fl_attack(cfg, round_idx, client_id, sample_client_num):
    attacker_ids = parse_attacker_ids(cfg.attack.attacker_id)
    if client_id not in attacker_ids:
        return False

    a3fl_cfg = get_a3fl_cfg(cfg)
    if not is_a3fl_active_round(cfg, round_idx):
        return False

    if a3fl_cfg.sample_method == 'random_updates':
        return client_id in get_a3fl_active_attacker_ids(cfg, round_idx,
                                                         sample_client_num)

    return True
