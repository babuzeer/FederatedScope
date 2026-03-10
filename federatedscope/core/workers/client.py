import copy
import logging
import sys
import pickle
import threading
import time

from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, \
    StandaloneDDPCommManager, gRPCCommManager
from federatedscope.core.monitors.early_stopper import EarlyStopper
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.secret_sharing import AdditiveSecretSharing
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    calculate_time_cost
from federatedscope.core.workers.base_client import BaseClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Client(BaseClient):
    """
    The Client class, which describes the behaviors of client in an FL \
    course. The behaviors are described by the handling functions (named as \
    ``callback_funcs_for_xxx``)

    Arguments:
        ID: The unique ID of the client, which is assigned by the server
        when joining the FL course
        server_id: (Default) 0
        state: The training round
        config: The configuration
        data: The data owned by the client
        model: The model maintained locally
        device: The device to run local training and evaluation

    Attributes:
        ID: ID of worker
        state: the training round index
        model: the model maintained locally
        cfg: the configuration of FL course, \
            see ``federatedscope.core.configs``
        mode: the run mode for FL, ``distributed`` or ``standalone``
        monitor: monite FL course and record metrics, \
            see ``federatedscope.core.monitors.monitor.Monitor``
        trainer: instantiated trainer, see ``federatedscope.core.trainers``
        best_results: best results ever seen
        history_results: all evaluation results
        early_stopper: determine when to early stop, \
            see ``federatedscope.core.monitors.early_stopper.EarlyStopper``
        ss_manager: secret sharing manager
        msg_buffer: dict buffer for storing message
        comm_manager: manager for communication, \
            see ``federatedscope.core.communication``
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
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(Client, self).__init__(ID, state, config, model, strategy)

        self.data = data

        # Register message handlers
        self._register_default_handlers()

        # Un-configured worker
        if config is None:
            return

        # the unseen_client indicates that whether this client contributes to
        # FL process by training on its local data and uploading the local
        # model update, which is useful for check the participation
        # generalization gap in
        # [ICLR'22, What Do We Mean by Generalization in Federated Learning?]
        self.is_unseen_client = is_unseen_client

        # Parse the attack_id since we support both 'int' (for single attack)
        # and 'list' (for multiple attacks) for config.attack.attack_id
        parsed_attack_ids = list()
        if isinstance(config.attack.attacker_id, int):
            parsed_attack_ids.append(config.attack.attacker_id)
        elif isinstance(config.attack.attacker_id, list):
            parsed_attack_ids = config.attack.attacker_id
        else:
            raise TypeError(f"The expected types of config.attack.attack_id "
                            f"include 'int' and 'list', but we got "
                            f"{type(config.attack.attacker_id)}")

        # Attack only support the stand alone model;
        # Check if is a attacker; a client is a attacker if the
        # config.attack.attack_method is provided
        self.is_attacker = ID in parsed_attack_ids and \
            config.attack.attack_method != '' and \
            config.federate.mode == 'standalone'

        # Build Trainer
        # trainer might need configurations other than those of trainer node
        self.trainer = get_trainer(model=model,
                                   data=data,
                                   device=device,
                                   config=self._cfg,
                                   is_attacker=self.is_attacker,
                                   monitor=self._monitor)
        print(type(self.trainer))
        self.device = device

        # For client-side evaluation
        self.best_results = dict()
        self.history_results = dict()
        # in local or global training mode, we do use the early stopper.
        # Otherwise, we set patience=0 to deactivate the local early-stopper
        patience = self._cfg.early_stop.patience if \
            self._cfg.federate.method in [
                "local", "global"
            ] else 0
        self.early_stopper = EarlyStopper(
            patience, self._cfg.early_stop.delta,
            self._cfg.early_stop.improve_indicator_mode,
            self._monitor.the_larger_the_better)

        # Secret Sharing Manager and message buffer
        self.ss_manager = AdditiveSecretSharing(
            shared_party_num=int(self._cfg.federate.sample_client_num
                                 )) if self._cfg.federate.use_ss else None
        self.msg_buffer = {'train': dict(), 'eval': dict()}

        # Communication and communication ability
        if 'resource_info' in kwargs and kwargs['resource_info'] is not None:
            self.comp_speed = float(
                kwargs['resource_info']['computation']) / 1000.  # (s/sample)
            self.comm_bandwidth = float(
                kwargs['resource_info']['communication'])  # (kbit/s)
        else:
            self.comp_speed = None
            self.comm_bandwidth = None

        if self._cfg.backend == 'torch':
            self.model_size = sys.getsizeof(pickle.dumps(
                self.model)) / 1024.0 * 8.  # kbits
        else:
            # TODO: calculate model size for TF Model
            self.model_size = 1.0
            logger.warning(f'The calculation of model size in backend:'
                           f'{self._cfg.backend} is not provided.')

        # Initialize communication manager
        self.server_id = server_id
        if self.mode == 'standalone':
            comm_queue = kwargs['shared_comm_queue']
            if self._cfg.federate.process_num <= 1:
                self.comm_manager = StandaloneCommManager(
                    comm_queue=comm_queue, monitor=self._monitor)
            else:
                self.comm_manager = StandaloneDDPCommManager(
                    comm_queue=comm_queue, monitor=self._monitor)
            self.local_address = None
        elif self.mode == 'distributed':
            host = kwargs['host']
            port = kwargs['port']
            server_host = kwargs['server_host']
            server_port = kwargs['server_port']
            self.comm_manager = gRPCCommManager(
                host=host,
                port=port,
                client_num=self._cfg.federate.client_num,
                cfg=self._cfg.distribute)
            logger.info('Client: Listen to {}:{}...'.format(host, port))
            self.comm_manager.add_neighbors(neighbor_id=server_id,
                                            address={
                                                'host': server_host,
                                                'port': server_port
                                            })
            self.local_address = {
                'host': self.comm_manager.host,
                'port': self.comm_manager.port
            }

        # Heartbeat sender for distributed mode
        self._heartbeat_stop_event = threading.Event()
        self._heartbeat_thread = None

        # Reconnection support for distributed mode
        self._reconnect_enabled = getattr(self._cfg.distribute,
                                          'reconnect_enabled', True)
        self._reconnect_stop_event = threading.Event()
        self._reconnect_thread = None
        self._connection_lost_event = threading.Event()
        self._is_reconnecting = False
        self._reconnect_lock = threading.Lock()

    def _start_heartbeat_sender(self):
        """
        Start a background thread that periodically sends heartbeat messages
        to the server to indicate the client is still alive.
        """
        if self.mode != 'distributed':
            return

        heartbeat_interval = getattr(self._cfg.distribute,
                                     'heartbeat_interval', 30)

        def _sender_loop():
            while not self._heartbeat_stop_event.is_set():
                # Wait for the interval (can be interrupted by stop event)
                if self._heartbeat_stop_event.wait(heartbeat_interval):
                    break  # Stop event was set

                # Only send heartbeat if we have a valid ID
                if self.ID != -1:
                    try:
                        self.comm_manager.send(
                            Message(msg_type='heartbeat',
                                    sender=self.ID,
                                    receiver=[self.server_id],
                                    state=self.state,
                                    timestamp=0,
                                    content='heartbeat'))
                        logger.debug(f'Client #{self.ID}: Sent heartbeat')
                    except Exception as e:
                        logger.warning(
                            f'Client #{self.ID}: Failed to send heartbeat: {e}'
                        )

        self._heartbeat_thread = threading.Thread(
            target=_sender_loop,
            name=f'HeartbeatSender-{self.ID}',
            daemon=True)
        self._heartbeat_thread.start()
        logger.info(f'Client: Heartbeat sender started '
                    f'(interval={heartbeat_interval}s)')

    def _stop_heartbeat_sender(self):
        """Stop the heartbeat sender thread."""
        if self._heartbeat_thread is not None:
            self._heartbeat_stop_event.set()
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None

    def _start_reconnect_monitor(self):
        """
        Start a background thread that monitors connection status and
        attempts to reconnect when connection is lost.
        """
        if self.mode != 'distributed' or not self._reconnect_enabled:
            return

        # Register callback for connection lost events
        if hasattr(self.comm_manager, 'register_connection_lost_callback'):
            self.comm_manager.register_connection_lost_callback(
                self._on_connection_lost)

        def _reconnect_loop():
            while not self._reconnect_stop_event.is_set():
                # Wait for connection lost event
                if self._connection_lost_event.wait(timeout=1.0):
                    if self._reconnect_stop_event.is_set():
                        break
                    self._connection_lost_event.clear()
                    self._attempt_reconnect()

        self._reconnect_thread = threading.Thread(
            target=_reconnect_loop,
            name=f'ReconnectMonitor-{self.ID}',
            daemon=True)
        self._reconnect_thread.start()
        logger.info('Client: Reconnect monitor started')

    def _on_connection_lost(self, neighbor_id):
        """Callback when connection to server is lost."""
        if neighbor_id == self.server_id:
            logger.warning(f'Client #{self.ID}: Connection to server lost, '
                           f'triggering reconnection...')
            self._connection_lost_event.set()

    def _attempt_reconnect(self):
        """
        Attempt to reconnect to the server using exponential backoff
        with jitter (following gRPC backoff algorithm).
        """
        with self._reconnect_lock:
            if self._is_reconnecting:
                return
            self._is_reconnecting = True

        max_attempts = getattr(self._cfg.distribute, 'reconnect_max_attempts',
                               10)
        initial_backoff = getattr(self._cfg.distribute,
                                  'reconnect_initial_backoff', 1.0)
        max_backoff = getattr(self._cfg.distribute, 'reconnect_max_backoff',
                              60.0)
        multiplier = getattr(self._cfg.distribute,
                             'reconnect_backoff_multiplier', 2.0)
        jitter = getattr(self._cfg.distribute, 'reconnect_jitter', 0.2)

        current_backoff = initial_backoff
        import random

        for attempt in range(max_attempts):
            if self._reconnect_stop_event.is_set():
                break

            logger.info(f'Client #{self.ID}: Reconnection attempt '
                        f'{attempt + 1}/{max_attempts}...')

            try:
                # Try to send a join_in message to reconnect
                self.comm_manager.mark_neighbor_online(self.server_id)
                self.comm_manager.send(
                    Message(msg_type='join_in',
                            sender=self.ID,
                            receiver=[self.server_id],
                            timestamp=0,
                            content=self.local_address))

                # Wait a bit and check if server responds
                time.sleep(2.0)

                # If we're still online (heartbeat ack received), reconnection
                # is successful
                if hasattr(self.comm_manager, 'is_neighbor_alive'):
                    if self.comm_manager.is_neighbor_alive(self.server_id):
                        logger.info(f'Client #{self.ID}: Reconnection '
                                    f'successful!')
                        with self._reconnect_lock:
                            self._is_reconnecting = False
                        return

            except Exception as e:
                logger.warning(f'Client #{self.ID}: Reconnection attempt '
                               f'{attempt + 1} failed: {e}')

            # Calculate next backoff with jitter
            jitter_value = random.uniform(-jitter, jitter) * current_backoff
            sleep_time = min(current_backoff + jitter_value, max_backoff)
            logger.debug(f'Client #{self.ID}: Waiting {sleep_time:.2f}s '
                         f'before next reconnection attempt')
            time.sleep(sleep_time)
            current_backoff = min(current_backoff * multiplier, max_backoff)

        logger.error(f'Client #{self.ID}: Failed to reconnect after '
                     f'{max_attempts} attempts')
        with self._reconnect_lock:
            self._is_reconnecting = False
        # Server is permanently unreachable; terminate the client process
        logger.error(f'Client #{self.ID}: Server unreachable, exiting.')
        import os
        os._exit(1)

    def _stop_reconnect_monitor(self):
        """Stop the reconnect monitor thread."""
        if self._reconnect_thread is not None:
            self._reconnect_stop_event.set()
            self._connection_lost_event.set()  # Unblock the wait
            self._reconnect_thread.join(timeout=5)
            self._reconnect_thread = None

    def _gen_timestamp(self, init_timestamp, instance_number):
        if init_timestamp is None:
            return None

        comp_cost, comm_cost = calculate_time_cost(
            instance_number=instance_number,
            comm_size=self.model_size,
            comp_speed=self.comp_speed,
            comm_bandwidth=self.comm_bandwidth)
        return init_timestamp + comp_cost + comm_cost

    def _calculate_model_delta(self, init_model, updated_model):
        if not isinstance(init_model, list):
            init_model = [init_model]
            updated_model = [updated_model]

        model_deltas = list()
        for model_index in range(len(init_model)):
            model_delta = copy.deepcopy(updated_model[model_index])
            for key in init_model[model_index].keys():
                if key not in updated_model[model_index].keys():
                    continue
                model_delta[key] = updated_model[model_index][
                    key] - init_model[model_index][key]
            model_deltas.append(model_delta)

        if len(model_deltas) > 1:
            return model_deltas
        else:
            return model_deltas[0]

    def join_in(self):
        """
        To send ``join_in`` message to the server for joining in the FL course.
        """
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    timestamp=0,
                    content=self.local_address))

    def run(self):
        """
        To listen to the message and handle them accordingly (used for \
        distributed mode)
        """
        # Start heartbeat sender for distributed mode
        if self.mode == 'distributed':
            self._start_heartbeat_sender()
            self._start_reconnect_monitor()

        # Buffer for messages received before ID assignment
        pending_messages = []

        while True:
            msg = self.comm_manager.receive()

            # In distributed mode, ensure ID is assigned before
            # processing training messages
            if self._mode == 'distributed' and self.ID == -1:
                if msg.msg_type == 'assign_client_id':
                    # Process ID assignment immediately
                    self.msg_handlers[msg.msg_type](msg)
                    # Process any pending messages after ID assignment
                    for pending_msg in pending_messages:
                        if self.state <= pending_msg.state:
                            self.msg_handlers[pending_msg.msg_type](
                                pending_msg)
                    pending_messages.clear()
                else:
                    # Buffer other messages until ID is assigned
                    pending_messages.append(msg)
                    if msg.msg_type == 'finish':
                        break
                    continue

            if self.state <= msg.state:
                self.msg_handlers[msg.msg_type](msg)

            if msg.msg_type == 'finish':
                break

    def run_standalone(self):
        """
        Run in standalone mode
        """
        self.join_in()
        self.run()

    def callback_funcs_for_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, \
        which triggers the local training process. \
        This handling function is widely used in various FL courses.

        Arguments:
            message: The received message
        """
        if 'ss' in message.msg_type:
            # A fragment of the shared secret
            state, content, timestamp = message.state, message.content, \
                                        message.timestamp
            self.msg_buffer['train'][state].append(content)

            if len(self.msg_buffer['train']
                   [state]) == self._cfg.federate.client_num:
                # Check whether the received fragments are enough
                model_list = self.msg_buffer['train'][state]
                sample_size, first_aggregate_model_para = model_list[0]
                single_model_case = True
                if isinstance(first_aggregate_model_para, list):
                    assert isinstance(first_aggregate_model_para[0], dict), \
                        "aggregate_model_para should a list of multiple " \
                        "state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(first_aggregate_model_para, dict), \
                        "aggregate_model_para should " \
                        "a state_dict for single model case"
                    first_aggregate_model_para = [first_aggregate_model_para]
                    model_list = [[model] for model in model_list]

                for sub_model_idx, aggregate_single_model_para in enumerate(
                        first_aggregate_model_para):
                    for key in aggregate_single_model_para:
                        for i in range(1, len(model_list)):
                            aggregate_single_model_para[key] += model_list[i][
                                sub_model_idx][key]

                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[self.server_id],
                            state=self.state,
                            timestamp=timestamp,
                            content=(sample_size, first_aggregate_model_para[0]
                                     if single_model_case else
                                     first_aggregate_model_para)))

        else:
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
            # ensure all the model params (which might be updated by other
            # clients in the previous local training process) are overwritten
            # and synchronized with the received model
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
                # for these cases (1) unseen client (2) isolated_global_mode,
                # we do not local train and upload local model
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

            # Return the feedbacks to the server after local update
            if self._cfg.federate.use_ss:
                assert not self.is_unseen_client, \
                    "Un-support using secret sharing for unseen clients." \
                    "i.e., you set cfg.federate.use_ss=True and " \
                    "cfg.federate.unseen_clients_rate in (0, 1)"
                single_model_case = True
                if isinstance(model_para_all, list):
                    assert isinstance(model_para_all[0], dict), \
                        "model_para should a list of " \
                        "multiple state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(model_para_all, dict), \
                        "model_para should a state_dict for single model case"
                    model_para_all = [model_para_all]
                model_para_list_all = []
                for model_para in model_para_all:
                    for key in model_para:
                        model_para[key] = model_para[key] * sample_size
                    model_para_list = self.ss_manager.secret_split(model_para)
                    model_para_list_all.append(model_para_list)
                frame_idx = 0
                for neighbor in self.comm_manager.neighbors:
                    if neighbor != self.server_id:
                        content_frame = model_para_list_all[0][frame_idx] if \
                            single_model_case else \
                            [model_para_list[frame_idx] for model_para_list
                             in model_para_list_all]
                        self.comm_manager.send(
                            Message(msg_type='ss_model_para',
                                    sender=self.ID,
                                    receiver=[neighbor],
                                    state=self.state,
                                    timestamp=self._gen_timestamp(
                                        init_timestamp=timestamp,
                                        instance_number=sample_size),
                                    content=content_frame))
                        frame_idx += 1
                content_frame = model_para_list_all[0][frame_idx] if \
                    single_model_case else \
                    [model_para_list[frame_idx] for model_para_list in
                     model_para_list_all]
                self.msg_buffer['train'][self.state] = [(sample_size,
                                                         content_frame)]
            else:
                if self._cfg.asyn.use or self._cfg.aggregator.robust_rule in \
                        ['krum', 'normbounding', 'median', 'trimmedmean',
                         'bulyan']:
                    # Return the model delta when using asynchronous training
                    # protocol, because the staled updated might be discounted
                    # and cause that the sum of the aggregated weights might
                    # not be equal to 1
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
                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self._gen_timestamp(
                                init_timestamp=timestamp,
                                instance_number=sample_size),
                            content=(sample_size, shared_model_para)))

    def callback_funcs_for_assign_id(self, message: Message):
        """
        The handling function for receiving the client_ID assigned by the \
        server (during the joining process), which is used in the \
        distributed mode.

        Arguments:
            message: The received message
        """
        content = message.content
        self.ID = int(content)
        logger.info('Client (address {}:{}) is assigned with #{:d}.'.format(
            self.comm_manager.host, self.comm_manager.port, self.ID))

    def callback_funcs_for_join_in_info(self, message: Message):
        """
        The handling function for receiving the request of join in \
        information (such as ``batch_size``, ``num_of_samples``) during \
        the joining process.

        Arguments:
            message: The received message
        """
        requirements = message.content
        timestamp = message.timestamp
        join_in_info = dict()
        for requirement in requirements:
            if requirement.lower() == 'num_sample':
                if self._cfg.train.batch_or_epoch == 'batch':
                    num_sample = self._cfg.train.local_update_steps * \
                                 self._cfg.dataloader.batch_size
                else:
                    num_sample = self._cfg.train.local_update_steps * \
                                 len(self.trainer.data.train_data)
                join_in_info['num_sample'] = num_sample
                if self._cfg.trainer.type == 'nodefullbatch_trainer':
                    join_in_info['num_sample'] = \
                        self.trainer.data.train_data.x.shape[0]
            elif requirement.lower() == 'client_resource':
                assert self.comm_bandwidth is not None and self.comp_speed \
                       is not None, "The requirement join_in_info " \
                                    "'client_resource' does not exist."
                join_in_info['client_resource'] = self.model_size / \
                    self.comm_bandwidth + self.comp_speed
            else:
                raise ValueError(
                    'Fail to get the join in information with type {}'.format(
                        requirement))
        self.comm_manager.send(
            Message(msg_type='join_in_info',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    timestamp=timestamp,
                    content=join_in_info))

    def callback_funcs_for_address(self, message: Message):
        """
        The handling function for receiving other clients' IP addresses, \
        which is used for constructing a complex topology

        Arguments:
            message: The received message
        """
        content = message.content
        for neighbor_id, address in content.items():
            if int(neighbor_id) != self.ID:
                self.comm_manager.add_neighbors(neighbor_id, address)

    def callback_funcs_for_evaluate(self, message: Message):
        """
        The handling function for receiving the request of evaluating

        Arguments:
            message: The received message
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)
        if self.early_stopper.early_stopped and self._cfg.federate.method in [
                "local", "global"
        ]:
            metrics = list(self.best_results.values())[0]
        else:
            metrics = {}
            if self._cfg.finetune.before_eval:
                self.trainer.finetune()
            for split in self._cfg.eval.split:
                # TODO: The time cost of evaluation is not considered here
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)

                if self._cfg.federate.mode == 'distributed':
                    logger.info(
                        self._monitor.format_eval_res(eval_metrics,
                                                      rnd=self.state,
                                                      role='Client #{}'.format(
                                                          self.ID),
                                                      return_raw=True))

                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                forms=['raw'],
                return_raw=True)
            self._monitor.update_best_result(self.best_results,
                                             formatted_eval_res['Results_raw'],
                                             results_type=f"client #{self.ID}")
            self.history_results = merge_dict_of_results(
                self.history_results, formatted_eval_res['Results_raw'])
            self.early_stopper.track_and_check(self.history_results[
                self._cfg.eval.best_res_update_round_wise_key])

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))

    def callback_funcs_for_finish(self, message: Message):
        """
        The handling function for receiving the signal of finishing the FL \
        course.

        Arguments:
            message: The received message
        """
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")

        # Stop heartbeat sender before finishing
        self._stop_heartbeat_sender()
        # Stop reconnect monitor before finishing
        self._stop_reconnect_monitor()

        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)

        self._monitor.finish_fl()

    def callback_funcs_for_converged(self, message: Message):
        """
        The handling function for receiving the signal that the FL course \
        converged

        Arguments:
            message: The received message
        """
        self._monitor.global_converged()

    def callback_funcs_for_heartbeat_ack(self, message: Message):
        """
        The handling function for receiving heartbeat acknowledgment from \
        the server. This confirms the connection is alive.

        Arguments:
            message: The received message
        """
        logger.debug(f'Client #{self.ID}: Received heartbeat ack from server')

    @classmethod
    def get_msg_handler_dict(cls):
        return cls().msg_handlers_str
