from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_attack_cfg(cfg):

    # ---------------------------------------------------------------------- #
    # attack
    # ---------------------------------------------------------------------- #
    cfg.attack = CN()
    cfg.attack.attack_method = ''
    # for gan_attack
    cfg.attack.target_label_ind = -1
    cfg.attack.attacker_id = -1
    # When enabled, server-side aggregation ignores statistics/prototypes and
    # model updates from clients listed in cfg.attack.attacker_id.
    cfg.attack.aggregate_benign_only = False

    # for backdoor attack

    cfg.attack.edge_path = 'edge_data/'
    cfg.attack.trigger_path = 'trigger/'
    cfg.attack.setting = 'fix'
    cfg.attack.freq = 10
    cfg.attack.insert_round = 100000
    cfg.attack.mean = [0.9637]
    cfg.attack.std = [0.1592]
    cfg.attack.trigger_type = 'edge'
    cfg.attack.label_type = 'dirty'
    # dirty, clean_label, dirty-label attack is all2one attack.
    cfg.attack.edge_num = 100
    cfg.attack.poison_ratio = 0.5
    cfg.attack.scale_poisoning = False
    cfg.attack.scale_para = 1.0
    cfg.attack.pgd_poisoning = False
    cfg.attack.pgd_lr = 0.1
    cfg.attack.pgd_eps = 2
    cfg.attack.self_opt = False
    cfg.attack.self_lr = 0.05
    cfg.attack.self_epoch = 6
    # Note: the mean and std should be the list type.

    # for A3FL
    cfg.attack.a3fl = CN()
    cfg.attack.a3fl.poison_epochs = 10
    cfg.attack.a3fl.sample_method = 'random'
    cfg.attack.a3fl.sample_poison_ratio = 0.2
    cfg.attack.a3fl.start_round = -1
    cfg.attack.a3fl.trigger_size = 5
    cfg.attack.a3fl.trigger_offset = 2
    cfg.attack.a3fl.trigger_init = 0.5
    cfg.attack.a3fl.trigger_outer_epochs = 20
    cfg.attack.a3fl.trigger_update_interval = 1
    cfg.attack.a3fl.trigger_search_batches = 2
    cfg.attack.a3fl.trigger_lr = 0.01
    cfg.attack.a3fl.trigger_clip_min = -2.0
    cfg.attack.a3fl.trigger_clip_max = 2.0
    cfg.attack.a3fl.save_trigger_samples = False
    cfg.attack.a3fl.save_trigger_max_samples = 4
    cfg.attack.a3fl.poison_feature_repeat = 1
    cfg.attack.a3fl.poison_train_epochs = 0
    cfg.attack.a3fl.poison_train_lr = 0.0
    cfg.attack.a3fl.update_scale = 1.0
    cfg.attack.a3fl.target_row_scale = 1.0

    # for label-flipping data poisoning on the GGEUR client
    cfg.attack.label_flip = CN()
    cfg.attack.label_flip.source_label_ind = -1
    cfg.attack.label_flip.target_label_ind = -1
    cfg.attack.label_flip.replacement_pairs = []
    cfg.attack.label_flip.all_to_target = False
    cfg.attack.label_flip.poison_ratio = 1.0
    cfg.attack.label_flip.start_round = -1
    cfg.attack.label_flip.poison_epochs = 0
    cfg.attack.label_flip.poison_statistics = True
    cfg.attack.label_flip.poison_training = True

    # for A Little Is Enough / ALIE model-poisoning attack on GGEUR
    # Malicious clients keep their local training unchanged; before server-side
    # aggregation, their submitted model parameters are replaced by
    # mean +/- z * std estimated per parameter dimension.
    cfg.attack.little_is_enough = CN()
    cfg.attack.little_is_enough.start_round = -1
    cfg.attack.little_is_enough.poison_epochs = 0
    cfg.attack.little_is_enough.z = 1.0
    cfg.attack.little_is_enough.auto_z = False
    cfg.attack.little_is_enough.max_z = 1.5
    cfg.attack.little_is_enough.direction = 'positive'
    cfg.attack.little_is_enough.stats_source = 'attacker'
    cfg.attack.little_is_enough.min_std = 1e-6
    cfg.attack.little_is_enough.min_attackers = 1
    cfg.attack.little_is_enough.target_models = ['mlp', 'classifier']
    cfg.attack.little_is_enough.log_detail = True
    cfg.attack.little_is_enough.log_norms = True

    # for CERBERUS backdoor attack on the GGEUR client
    cfg.attack.cerberus = CN()
    cfg.attack.cerberus.start_round = -1
    cfg.attack.cerberus.poison_epochs = 10
    cfg.attack.cerberus.alpha_loss = 0.01
    cfg.attack.cerberus.beta_loss = 0.01
    cfg.attack.cerberus.clean_anchor_lr = 0.001
    cfg.attack.cerberus.clean_anchor_epochs = 1
    cfg.attack.cerberus.shadow_lr = 0.001
    cfg.attack.cerberus.shadow_epochs = 1
    cfg.attack.cerberus.poison_lr = 0.001
    cfg.attack.cerberus.poison_optimizer = 'Adam'
    cfg.attack.cerberus.internal_poison_epochs = 1
    cfg.attack.cerberus.poisoning_per_batch = 0
    cfg.attack.cerberus.preserve_clean_batches = False
    cfg.attack.cerberus.trigger_init = 0.5
    cfg.attack.cerberus.pattern_size = 4
    cfg.attack.cerberus.pattern_offset = 0
    cfg.attack.cerberus.poison_pattern = []
    cfg.attack.cerberus.share_model_meta = False
    cfg.attack.cerberus.aggregate_benign_only = False
    cfg.attack.cerberus.prefer_best_trigger = True
    cfg.attack.cerberus.best_trigger_min_improvement = 1e-4
    cfg.attack.cerberus.eval_poison_on_local_model = False
    cfg.attack.cerberus.eval_split = 'test'
    cfg.attack.cerberus.clean_ce_weight = 1.0
    cfg.attack.cerberus.poison_ce_weight = 1.0
    cfg.attack.cerberus.clean_target_suppression_weight = 0.0
    cfg.attack.cerberus.clean_target_margin = 0.5
    cfg.attack.cerberus.anchor_logits_weight = 0.0
    cfg.attack.cerberus.backdoor_retain_weight = 0.0
    cfg.attack.cerberus.replay_as_poison_augmentation = False
    cfg.attack.cerberus.use_trigger_replay = False
    cfg.attack.cerberus.retain_current_when_no_replay = False
    cfg.attack.cerberus.constrain_update_to_anchor = False
    cfg.attack.cerberus.anchor_residual_gamma = 0.5
    cfg.attack.cerberus.benign_reference_lr = 0.001
    cfg.attack.cerberus.benign_reference_epochs = 0
    cfg.attack.cerberus.auto_scale = False
    cfg.attack.cerberus.auto_scale_max = 1.0
    cfg.attack.cerberus.scale_cap = 1.0
    cfg.attack.cerberus.trigger_search_steps = 0
    cfg.attack.cerberus.trigger_update_interval = 1
    cfg.attack.cerberus.trigger_search_batches = 2
    cfg.attack.cerberus.trigger_search_batch_size = 8
    cfg.attack.cerberus.trigger_search_lr = 0.05
    cfg.attack.cerberus.trigger_search_clip_min = -2.5
    cfg.attack.cerberus.trigger_search_clip_max = 2.5
    cfg.attack.cerberus.trigger_search_proj_norm = 12.0
    cfg.attack.cerberus.trigger_search_target_margin = 1.0
    cfg.attack.cerberus.trigger_search_gain_weight = 0.5
    cfg.attack.cerberus.trigger_search_gain_margin = 0.5
    cfg.attack.cerberus.trigger_search_l2_weight = 0.0001

    # for SABRE full-image additive backdoor on the GGEUR MLP head
    cfg.attack.sabre = CN()
    cfg.attack.sabre.start_round = -1
    cfg.attack.sabre.poison_epochs = 10
    cfg.attack.sabre.clean_anchor_lr = 0.001
    cfg.attack.sabre.clean_anchor_epochs = 1
    cfg.attack.sabre.poison_lr = 0.001
    cfg.attack.sabre.poison_optimizer = 'Adam'
    cfg.attack.sabre.internal_poison_epochs = 1
    cfg.attack.sabre.poisoning_per_batch = 0
    cfg.attack.sabre.preserve_clean_batches = True
    cfg.attack.sabre.trigger_init_mode = 'uniform'
    cfg.attack.sabre.trigger_init = 0.0
    cfg.attack.sabre.trigger_seed = 0
    cfg.attack.sabre.trigger_random_scale = 0.01
    cfg.attack.sabre.max_poison_samples = 0
    cfg.attack.sabre.poison_feature_repeat = 1
    cfg.attack.sabre.clean_ce_weight = 1.0
    cfg.attack.sabre.poison_ce_weight = 1.0
    cfg.attack.sabre.anchor_loss_weight = 0.01
    cfg.attack.sabre.clean_target_suppression_weight = 0.0
    cfg.attack.sabre.clean_target_margin = 0.5
    cfg.attack.sabre.constrain_update_to_anchor = False
    cfg.attack.sabre.anchor_residual_gamma = 0.5
    cfg.attack.sabre.image_clip_min = -3.0
    cfg.attack.sabre.image_clip_max = 3.0
    cfg.attack.sabre.trigger_search_steps = 0
    cfg.attack.sabre.trigger_update_interval = 1
    cfg.attack.sabre.trigger_search_batches = 2
    cfg.attack.sabre.trigger_search_batch_size = 8
    cfg.attack.sabre.trigger_search_lr = 0.01
    cfg.attack.sabre.trigger_search_clip_min = -0.05
    cfg.attack.sabre.trigger_search_clip_max = 0.05
    cfg.attack.sabre.trigger_search_proj_norm = 4.0
    cfg.attack.sabre.trigger_search_target_margin = 1.0
    cfg.attack.sabre.trigger_search_gain_weight = 0.5
    cfg.attack.sabre.trigger_search_gain_margin = 0.5
    cfg.attack.sabre.trigger_search_l2_weight = 0.0001

    # for reconstruct_opt
    cfg.attack.reconstruct_lr = 0.01
    cfg.attack.reconstruct_optim = 'Adam'
    cfg.attack.info_diff_type = 'l2'
    cfg.attack.max_ite = 400
    cfg.attack.alpha_TV = 0.001

    # for active PIA attack
    cfg.attack.alpha_prop_loss = 0

    # for passive PIA attack
    cfg.attack.classifier_PIA = 'randomforest'

    # for gradient Ascent --- MIA attack
    cfg.attack.inject_round = 0
    cfg.attack.mia_simulate_in_round = 20
    cfg.attack.mia_is_simulate_in = False

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_attack_cfg)


def assert_attack_cfg(cfg):
    pass


register_config("attack", extend_attack_cfg)
