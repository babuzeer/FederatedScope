def wrap_attacker_trainer(base_trainer, config):
    '''Wrap the trainer for attack client.
    Args:
        base_trainer (core.trainers.GeneralTorchTrainer): the trainer that
        will be wrapped;
        config (federatedscope.core.configs.config.CN): the configure;

    :returns:
        The wrapped trainer; Type: core.trainers.GeneralTorchTrainer

    '''
    attack_method = config.attack.attack_method.lower()
    if attack_method == 'cerberus':
        # GGEUR implements CERBERUS inside its custom client training path.
        # No generic trainer wrapper is needed here.
        return base_trainer
    if attack_method == 'gan_attack':
        from federatedscope.attack.trainer import wrap_GANTrainer
        return wrap_GANTrainer(base_trainer)
    elif attack_method == 'gradascent':
        from federatedscope.attack.trainer import wrap_GradientAscentTrainer
        return wrap_GradientAscentTrainer(base_trainer)
    elif attack_method == 'backdoor':
        from federatedscope.attack.trainer import wrap_backdoorTrainer
        return wrap_backdoorTrainer(base_trainer)
    elif attack_method == 'a3fl':
        from federatedscope.attack.trainer import wrap_A3FLTrainer
        return wrap_A3FLTrainer(base_trainer)
    elif attack_method == 'gaussian_noise':
        from federatedscope.attack.trainer import wrap_GaussianAttackTrainer
        return wrap_GaussianAttackTrainer(base_trainer)
    else:
        raise ValueError('Trainer {} is not provided'.format(
            config.attack.attack_method))
