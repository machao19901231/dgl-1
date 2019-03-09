# This file contains distributed samplers.
from ...network import _send_subgraph, _recv_subgraph
from ...network import _batch_send_subgraph, _batch_recv_subgraph
from ...network import _create_sampler_sender, _create_sampler_receiver
from ...network import _finalize_sampler_sender, _finalize_sampler_receiver

class SamplerSender(object):
    """The SamplerSender class for DGL distributed sampler.

    Users use this class to send sampled subgraph to remote trainer.

    Parameters
    ----------
    ip : str
        ip address of remote trainer machine.
    port : int
        port of remote trainer machine.
    """
    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        self._sender = _create_sampler_sender(ip, port)

    def __del__(self):
        """Finalize Sender
        """
        # _finalize_sampler_sender will send a special message
        # to tell the remote trainer it has finished its job.
        _finalize_sampler_sender(self._sender)

    def Send(self, nodeflow):
        """Send sampled subgraph (NodeFlow) to remote trainer.

        Parameters
        ----------
        nodeflow : NodeFlow
            sampled NodeFlow object.
        """
        _send_subgraph(self._sender, nodeflow)

    def BatchSend(self, nodeflow_list):
        """Send a batch of sampled subgraph (NodeFlow) to remote trainer.

        Parameters
        ----------
        nodeflow_list : list
            a list of NodeFlow objects.
        """
        _batch_send_subgraph(self._sender, nodeflow_list)

class SamplerReceiver(object):
    """The SamplerReceiver class for DGL distributed sampler.

    Users use this class to receive sampled subgraph from remote samplers.

    Parameters
    ----------
    ip : str
        ip address of trainer machine.
    port : int
        listen port of trainer machine.
    num_sender : int
        total number of sampler nodes, use 1 by default. 
        SamplerReceiver can recv message from multiple senders concurrently.
    queue_size : int
        size (bytes) of message queue, use 200 MB by default.
    """
    def __init__(self, ip, port, num_sender=1, queue_size=200*1024*1024):
        self._ip = ip
        self._port = port
        self._num_sender = num_sender
        self._queue_size = queue_size
        self._receiver = _create_sampler_receiver(ip, port, num_sender, queue_size)

    def __del__(self):
        """Finalize Receiver
        """
        _finalize_sampler_receiver(self._receiver)

    def Receive(self):
        """Receive a NodeFlow object from remote sampler.

        Returns
        -------
        NodeFlow
            Sampled NodeFlow object
        """
        return _recv_subgraph(self._receiver)

    def BatchReceive(self):
        """Receive a batch of NodeFlow objects from remote sampler.

        Returns
        -------
        list
            A list of sampled NodeFlow object
        """
        return _batch_recv_subgraph(self._receiver)
