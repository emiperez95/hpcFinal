from helpers.singleton import Singleton
from distribution.comms_manager import CommsManager

@Singleton
class TopologyManager():
    '''
    node_topology: dict de nodos, con set de pu
    pu_info: dict de pu, con la informacion relevante
    active_pu: set de pu activos
    inactive_pu: set de pu inactivos
    offline_pu: set de pu fuera de linea
    '''

    def __init__(self):
        comms = CommsManager.instance()

        self.active_pu = set([])
        self.inactive_pu = set([])
        self.offline_pu = set([])
        
        self.pu_info = {}
        self.node_topology = {}
        for i, node in enumerate(comms.nodes_info):
            # pu_info load
            if i == comms.root:
                self.active_pu.add(i)
                self.pu_info[i] = {
                    "info" : node,
                    "role" : "root"
                }
            else:
                self.inactive_pu.add(i)
                self.pu_info[i] = {
                    "info" : node,
                    "role" : "none"
                }
            
            # node_topology
            if node["node"] not in self.node_topology:
                self.node_topology[node["node"]] = set([])
            self.node_topology[node["node"]].add(i)
        # print(self.pu_info)

    def get_node_topology(self):
        return self.node_topology

    def all_pu_from_node(self, node):
        return self.node_topology["node"]

    def inactive_workers_left(self):
        return len(self.inactive_pu) > 0

    # Return a worker from the node with the least work effort
    def get_best_worker(self):
        if self.inactive_workers_left():
            node_work_effort = self.__node_work_effort()
            min_node = None
            min_val = None
            for node in node_work_effort:
                if min_val == None or min_val > node[1]:
                    min_node = node[0]
                    min_val = node[1]

            return self.__inactive_pu(min_node)

    def assign_worker(self, worker):
        self.active_pu.add(worker)
        self.inactive_pu = self.inactive_pu - set([worker])
        self.pu_info[worker]["role"] = "slave"

    # Return the work effort of each node
    def __node_work_effort(self):
        node_list = self.__node_list()
        work_effort_list = []
        for node in node_list:
            work_effort = 0
            for pu in self.node_topology[node]:
                if pu in self.active_pu:
                    work_effort += 1
            work_effort_list.append(work_effort)
        return zip(node_list, work_effort_list)

    def __node_list(self):
        return [node_name for node_name in self.node_topology]

    # Return an inactive pu from node
    def __inactive_pu(self, node):
        for pu in self.node_topology[node]:
            if pu in self.inactive_pu:
                return pu

    def get_worker_pu(self):
        return self.active_pu - set([CommsManager.instance().root])