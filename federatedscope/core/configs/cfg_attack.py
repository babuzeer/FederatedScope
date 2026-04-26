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
    cfg.attack.text_trigger = ''
    cfg.attack.text_trigger_position = ''
    cfg.attack.scale_poisoning = False
    cfg.attack.scale_para = 1.0
    cfg.attack.pgd_poisoning = False
    cfg.attack.pgd_lr = 0.1
    cfg.attack.pgd_eps = 2
    cfg.attack.self_opt = False
    cfg.attack.self_lr = 0.05
    cfg.attack.self_epoch = 6
    # Note: the mean and std should be the list type.

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

    # for CerP / PFedBA (feature-space backdoor attack for GGEUR-style
    # pipelines)
    cfg.attack.cerp = CN()
    cfg.attack.cerp.start_round = 1
    cfg.attack.cerp.trigger_lr = 0.1
    cfg.attack.cerp.trigger_steps = 1
    cfg.attack.cerp.trigger_tune_batches = 4
    cfg.attack.cerp.trigger_init_scale = 0.02
    cfg.attack.cerp.trigger_max_norm = 1.0
    cfg.attack.cerp.trigger_space = 'feature'
    cfg.attack.cerp.trigger_text = 'cf mn bb tq'
    cfg.attack.cerp.text_batch_size = 32
    cfg.attack.cerp.lambda_model = 1e-4
    cfg.attack.cerp.lambda_similarity = 1e-4
    cfg.attack.cerp.lambda_trigger_reg = 1e-3
    cfg.attack.cerp.force_attacker_participation = False
    cfg.attack.cerp.eval_poison = True

    # PFedBA keeps the same knobs as CerP; this explicit block allows
    # `attack.attack_method: pfedba` with `attack.pfedba.*` config fields.
    cfg.attack.pfedba = CN()
    cfg.attack.pfedba.start_round = cfg.attack.cerp.start_round
    cfg.attack.pfedba.trigger_lr = cfg.attack.cerp.trigger_lr
    cfg.attack.pfedba.trigger_steps = cfg.attack.cerp.trigger_steps
    cfg.attack.pfedba.trigger_tune_batches = cfg.attack.cerp.trigger_tune_batches
    cfg.attack.pfedba.trigger_init_scale = cfg.attack.cerp.trigger_init_scale
    cfg.attack.pfedba.trigger_max_norm = cfg.attack.cerp.trigger_max_norm
    cfg.attack.pfedba.trigger_space = cfg.attack.cerp.trigger_space
    cfg.attack.pfedba.trigger_text = cfg.attack.cerp.trigger_text
    cfg.attack.pfedba.text_batch_size = cfg.attack.cerp.text_batch_size
    cfg.attack.pfedba.lambda_model = cfg.attack.cerp.lambda_model
    cfg.attack.pfedba.lambda_similarity = cfg.attack.cerp.lambda_similarity
    cfg.attack.pfedba.lambda_trigger_reg = cfg.attack.cerp.lambda_trigger_reg
    cfg.attack.pfedba.force_attacker_participation = \
        cfg.attack.cerp.force_attacker_participation
    cfg.attack.pfedba.eval_poison = cfg.attack.cerp.eval_poison

    # Bad-PFL uses a shared feature-space generator together with
    # disruptive noise, and runs after GGEUR feature augmentation.
    cfg.attack.bad_pfl = CN()
    cfg.attack.bad_pfl.start_round = 1
    cfg.attack.bad_pfl.generator_hidden_dim = 512
    cfg.attack.bad_pfl.generator_lr = 1e-3
    cfg.attack.bad_pfl.generator_steps = 1
    cfg.attack.bad_pfl.generator_tune_batches = 4
    cfg.attack.bad_pfl.trigger_scale = 0.2
    cfg.attack.bad_pfl.trigger_max_norm = 1.0
    cfg.attack.bad_pfl.disruptive_eps = 0.2
    cfg.attack.bad_pfl.disruptive_alpha = 0.2
    cfg.attack.bad_pfl.disruptive_steps = 1
    cfg.attack.bad_pfl.lambda_trigger_reg = 0.0
    cfg.attack.bad_pfl.clean_weight = 1.0
    cfg.attack.bad_pfl.poison_weight = 1.0
    cfg.attack.bad_pfl.append_poisoned = False
    cfg.attack.bad_pfl.poison_repeats = 1
    cfg.attack.bad_pfl.target_feature_blend = 0.0
    cfg.attack.bad_pfl.target_align_weight = 0.0
    cfg.attack.bad_pfl.margin_weight = 0.0
    cfg.attack.bad_pfl.use_shared_target_reference = True
    cfg.attack.bad_pfl.force_attacker_participation = False
    cfg.attack.bad_pfl.eval_poison = True

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_attack_cfg)


def assert_attack_cfg(cfg):
    pass


register_config("attack", extend_attack_cfg)
