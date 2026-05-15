import queue
import time
from collections import deque

from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc


class gRPCComServeFunc(gRPC_comm_manager_pb2_grpc.gRPCComServeFuncServicer):
    def __init__(self):
        self.msg_queue = deque()

    def sendMessage(self, request, context):
        self.msg_queue.append(request)

        return gRPC_comm_manager_pb2.MessageResponse(msg='ACK')

    def receive(self, timeout=None, poll_interval=0.1):
        start_time = time.time()
        while len(self.msg_queue) == 0:
            if timeout is not None and time.time() - start_time >= timeout:
                raise queue.Empty()
            time.sleep(poll_interval)
        msg = self.msg_queue.popleft()
        return msg
