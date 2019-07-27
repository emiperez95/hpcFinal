import logging
from threading import Thread

from distribution.node_client import NodeClient
from distribution.comms_manager import CommsManager
from helpers.topology import TopologyManager

_logger = logging.getLogger(__name__)

HEARTBEAT_FREQUENCY_SEC = 60


class Heartbeat(Thread):
    def __init__(self, event, kill_clients_on_disconnect):
        Thread.__init__(self)
        self.kill_clients_on_disconnect = kill_clients_on_disconnect
        self.stopped = event
        self.success = None
        self.node_client = CommsManager.instance()

    def run(self):
        while not self.stopped.wait(HEARTBEAT_FREQUENCY_SEC):
            active_clients = list(TopologyManager.instance().get_worker_pu())
            client_statuses = self.node_client.get_client_statuses(active_clients)
            dead_clients = [c for c in client_statuses if not c['alive'] or not c['busy']]
            alive_clients = [c for c in client_statuses if c['alive'] and c['busy']]

            if dead_clients and self.kill_clients_on_disconnect:
                printable_names = '.'.join([c['address'] for c in dead_clients])
                _logger.critical('Heartbeat: One or more clients ({}) are not alive anymore; '
                                 'exiting others as well.'.format(printable_names))

                self.node_client.stop_running_experiments(dead_clients)
                self.success = False
                return
            elif all(c['finished'] for c in alive_clients):
                _logger.info('Heartbeat: All clients finished their experiments.')
                self.success = True
                return
