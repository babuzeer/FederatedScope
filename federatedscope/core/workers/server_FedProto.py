"""
FedProto (Federated Prototypical Learning) Server Implementation

Reference:
FedProto: Federated Prototype Learning across Heterogeneous Clients

Core Functionality:
1. Prototype Aggregation: Aggregate local prototypes from clients
2. Global Prototype Computation: Compute weighted or average global prototypes
3. Prototype Distribution: Broadcast global prototypes to clients
4. Model Aggregation: Standard FedAvg for model parameters
"""

import logging
import torch
from federatedscope.core.workers import Server
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


class FedProtoServer(Server):
    """
    FedProto Server Implementation

    Responsibilities:
    1. Aggregate local prototypes from clients
    2. Compute global prototypes (mean or weighted mean)
    3. Distribute global prototypes to clients
    4. Perform standard model aggregation (FedAvg)

    Algorithm Flow (each communication round):
    1. Receive model parameters and local prototypes from clients
    2. Aggregate model parameters using FedAvg
    3. Aggregate prototypes to form global prototypes
    4. Broadcast updated model and global prototypes to clients
    """

    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):
        super(FedProtoServer, self).__init__(
            ID=ID,
            state=state,
            config=config,
            data=data,
            model=model,
            client_num=client_num,
            total_round_num=total_round_num,
            device=device,
            strategy=strategy,
            unseen_clients_id=unseen_clients_id,
            **kwargs
        )

        # FedProto specific parameters
        self.num_classes = config.model.num_classes
        self.embedding_dim = config.fedproto.embedding_dim
        self.aggregation_method = config.fedproto.aggregation_method

        # Storage for received prototypes
        self.received_prototypes = {}  # {client_id: (prototypes, sample_count)}

        # Global prototypes
        self.global_prototypes = None

        logger.info(f"FedProtoServer initialized: num_classes={self.num_classes}, "
                   f"embedding_dim={self.embedding_dim}, "
                   f"aggregation={self.aggregation_method}")

    def callback_funcs_model_para(self, message: Message):
        """
        Receive model updates and (optionally) local prototypes from clients.

        Clients upload a dict:
          {
            'sample_count': int,
            'model_para': state_dict (or list of state_dict for multi-model),
            'prototypes': Tensor[num_classes, embedding_dim] (optional)
          }

        This handler converts the payload back to the standard
        `(sample_size, model_para)` format used by the vanilla Server, while
        separately storing prototypes for server-side aggregation.
        """
        if self.is_finish:
            return 'finish'

        rnd = message.state
        sender = message.sender
        timestamp = message.timestamp
        payload = message.content
        self.sampler.change_state(sender, 'idle')

        # update the currency timestamp according to the received message
        assert timestamp >= self.cur_timestamp  # for test
        self.cur_timestamp = timestamp

        # Parse payload
        if isinstance(payload, dict):
            sample_size = int(payload.get('sample_count', 0) or 0)
            model_para = payload.get('model_para', {})
            prototypes = payload.get('prototypes', None)
            if rnd == self.state and prototypes is not None:
                self.received_prototypes[sender] = (prototypes, sample_size)
        else:
            # Fallback to vanilla format: (sample_size, model_para)
            sample_size, model_para = payload

        # dequantization
        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_dequantization
            if isinstance(model_para, list):  # multiple model
                model_para = [
                    symmetric_uniform_dequantization(x) for x in model_para
                ]
            else:
                model_para = symmetric_uniform_dequantization(model_para)

        content = (sample_size, model_para)

        if rnd == self.state:
            if rnd not in self.msg_buffer['train']:
                self.msg_buffer['train'][rnd] = dict()
            self.msg_buffer['train'][rnd][sender] = content
        elif rnd >= self.state - self.staleness_toleration:
            self.staled_msg_buffer.append((rnd, sender, content))
        else:
            logger.info(f'Drop a out-of-date message from round #{rnd}')
            self.dropout_num += 1

        if self._cfg.federate.online_aggr:
            self.aggregator.inc(content)

        move_on_flag = self.check_and_move_on()
        if self._cfg.asyn.use and self._cfg.asyn.broadcast_manner == \
                'after_receiving':
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=1)

        return move_on_flag

    def _perform_federated_aggregation(self):
        aggregated_num = super()._perform_federated_aggregation()
        self.aggregate_prototypes()
        return aggregated_num

    def aggregate_prototypes(self):
        """
        Aggregate local prototypes to form global prototypes

        Two aggregation methods:
        1. 'mean': Simple average of all prototypes
        2. 'weighted_mean': Weighted average based on number of samples per client

        Returns:
            torch.Tensor: Global prototypes of shape (num_classes, embedding_dim)
        """
        if len(self.received_prototypes) == 0:
            logger.warning("No prototypes received for aggregation")
            return

        # Collect all prototypes and weights
        all_prototypes = []
        all_weights = []

        for client_id, (prototypes, sample_count) in self.received_prototypes.items():
            # Move to device
            if isinstance(prototypes, torch.Tensor):
                prototypes = prototypes.to(self.device)
            else:
                prototypes = torch.tensor(prototypes, device=self.device)

            all_prototypes.append(prototypes)
            all_weights.append(sample_count)

        # Stack prototypes: (num_clients, num_classes, embedding_dim)
        stacked_prototypes = torch.stack(all_prototypes, dim=0)

        # Aggregate based on method
        if self.aggregation_method == 'mean':
            # Simple average
            self.global_prototypes = stacked_prototypes.mean(dim=0)
            logger.info(f"Aggregated prototypes using simple mean from {len(all_prototypes)} clients")

        elif self.aggregation_method == 'weighted_mean':
            # Weighted average based on sample counts
            weights = torch.tensor(all_weights, dtype=torch.float32, device=self.device)
            weights = weights / weights.sum()  # Normalize

            # Reshape for broadcasting: (num_clients, 1, 1)
            weights = weights.view(-1, 1, 1)

            # Weighted sum
            self.global_prototypes = (stacked_prototypes * weights).sum(dim=0)
            logger.info(f"Aggregated prototypes using weighted mean from {len(all_prototypes)} clients, "
                       f"total samples={sum(all_weights)}")

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Log statistics
        proto_norm = torch.norm(self.global_prototypes, p=2, dim=1).mean()
        logger.info(f"Global prototypes computed: shape={self.global_prototypes.shape}, "
                   f"avg_norm={proto_norm.item():.4f}")

        # Clear received prototypes for next round
        self.received_prototypes.clear()

    def broadcast_model_para(self,
                            msg_type='model_para',
                            sample_client_num=-1,
                            filter_unseen_clients=True):
        """
        Broadcast model parameters and (if available) global prototypes.

        To ensure clients train with the latest prototypes, prototypes are
        sent *before* the `model_para` message in each round.
        """
        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        # We define the evaluation happens at the end of an epoch
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state
        state_to_send = min(rnd, self.total_round_num)

        if self.global_prototypes is not None and msg_type == 'model_para':
            self.comm_manager.send(
                Message(msg_type='global_prototypes',
                        sender=self.ID,
                        receiver=receiver,
                        state=state_to_send,
                        timestamp=self.cur_timestamp,
                        content=self.global_prototypes.detach().cpu())
            )

        if self._noise_injector is not None and msg_type == 'model_para':
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        skip_broadcast = self._cfg.federate.method in ["local", "global"]
        if self.model_num > 1:
            model_para = [{} if skip_broadcast else model.state_dict()
                          for model in self.models]
        else:
            model_para = {} if skip_broadcast else self.models[0].state_dict()

        # quantization
        if msg_type == 'model_para' and not skip_broadcast and \
                self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_quantization
            nbits = self._cfg.quantization.nbits
            if self.model_num > 1:
                model_para = [
                    symmetric_uniform_quantization(x, nbits)
                    for x in model_para
                ]
            else:
                model_para = symmetric_uniform_quantization(model_para, nbits)

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=state_to_send,
                    timestamp=self.cur_timestamp,
                    content=model_para))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'seen')
