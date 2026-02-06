"""
FedProto Client Worker Implementation

Handles client-side operations for FedProto:
1. Receive global prototypes from server
2. Train local model with prototype-based loss
3. Compute and send local prototypes to server
4. Send updated model parameters
"""

import logging
from federatedscope.core.workers import Client
from federatedscope.core.message import Message
from federatedscope.register import register_worker

logger = logging.getLogger(__name__)


class FedProtoClient(Client):
    """
    FedProto Client Implementation

    Extends the standard client to:
    1. Receive and store global prototypes from server
    2. Update trainer with global prototypes
    3. Extract local prototypes after training
    4. Send both model parameters and prototypes to server
    """

    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):
        super(FedProtoClient, self).__init__(
            ID=ID,
            server_id=server_id,
            state=state,
            config=config,
            data=data,
            model=model,
            device=device,
            strategy=strategy,
            *args,
            **kwargs
        )

        # Register callback for global prototypes
        self.register_handlers('global_prototypes',
                              self.callback_funcs_for_global_prototypes)

        logger.info(f"FedProtoClient {self.ID} initialized")

    def callback_funcs_for_global_prototypes(self, message: Message):
        """
        Callback to receive global prototypes from server

        Args:
            message: Message containing global prototypes
        """
        global_prototypes = message.content
        logger.info(f"Client {self.ID} received global prototypes: shape={global_prototypes.shape}")

        # Update trainer with global prototypes
        if hasattr(self.trainer, 'update_global_prototypes'):
            self.trainer.update_global_prototypes(global_prototypes)
            logger.debug(f"Client {self.ID} updated trainer with global prototypes")
        else:
            logger.warning(f"Client {self.ID} trainer does not support global prototypes")

    def callback_funcs_for_model_para(self, message: Message):
        """
        Override to include local prototypes in uploads.
        """
        if 'ss' in message.msg_type:
            # Keep the secret-sharing fragment handling from the base client
            return super().callback_funcs_for_model_para(message)

        import copy

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        # dequantization
        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_dequantization
            if isinstance(content, list):  # multiple model
                content = [
                    symmetric_uniform_dequantization(x) for x in content
                ]
            else:
                content = symmetric_uniform_dequantization(content)

        # When clients share the local model, we must set strict=True to
        # ensure all the model params are overwritten and synchronized
        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device)

        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model)
        self.state = round

        skip_train_isolated_or_global_mode = \
            self.early_stopper.early_stopped and \
            self._cfg.federate.method in ["local", "global"]
        if self.is_unseen_client or skip_train_isolated_or_global_mode:
            sample_size, model_para_all, results = \
                0, self.trainer.get_model_para(), {}
            if skip_train_isolated_or_global_mode:
                logger.info(
                    f"[Local/Global mode] Client #{self.ID} has been "
                    f"early stopped, we will skip the local training")
                self._monitor.local_converged()
        else:
            if self.early_stopper.early_stopped and \
                    self._monitor.local_convergence_round == 0:
                logger.info(
                    f"[Normal FL Mode] Client #{self.ID} has been locally "
                    f"early stopped. "
                    f"The next FL update may result in negative effect")
                self._monitor.local_converged()

            sample_size, model_para_all, results = self.trainer.train()
            if self._cfg.federate.share_local_model and not \
                    self._cfg.federate.online_aggr:
                model_para_all = copy.deepcopy(model_para_all)

            train_log_res = self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                return_raw=True)
            logger.info(train_log_res)
            if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                self._monitor.save_formatted_results(train_log_res,
                                                     save_file_name="")

        if self._cfg.federate.use_ss:
            # FedProto currently does not support secret sharing uploads with prototypes
            return super().callback_funcs_for_model_para(message)

        if self._cfg.asyn.use or self._cfg.aggregator.robust_rule in \
                ['krum', 'normbounding', 'median', 'trimmedmean',
                 'bulyan']:
            shared_model_para = self._calculate_model_delta(
                init_model=content, updated_model=model_para_all)
        else:
            shared_model_para = model_para_all

        # quantization
        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_quantization
            nbits = self._cfg.quantization.nbits
            if isinstance(shared_model_para, list):
                shared_model_para = [
                    symmetric_uniform_quantization(x, nbits)
                    for x in shared_model_para
                ]
            else:
                shared_model_para = symmetric_uniform_quantization(
                    shared_model_para, nbits)

        local_prototypes = None
        if hasattr(self.trainer, 'get_local_prototypes'):
            local_prototypes = self.trainer.get_local_prototypes()

        send_content = {
            'sample_count': sample_size,
            'model_para': shared_model_para,
        }
        if local_prototypes is not None:
            send_content['prototypes'] = local_prototypes.detach().cpu()

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=self._gen_timestamp(
                        init_timestamp=timestamp,
                        instance_number=sample_size),
                    content=send_content))



def call_fedproto_worker(method):
    if method.lower() == 'fedproto':
        from federatedscope.core.workers.server_FedProto import FedProtoServer
        return {
            'client': FedProtoClient,
            'server': FedProtoServer
        }
    return None


register_worker('fedproto', call_fedproto_worker)
