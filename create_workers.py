import socket
import fcntl
import struct
import time
import ray
import sys
import os


def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,
        struct.pack('256s', bytes(ifname[:15], 'utf-8'))
    )[20:24])


worker_task = """
ray start --redis-address={}
sleep {}
"""


@ray.remote
def get_workers():
    time.sleep(1e-4)
    return ray.services.get_node_ip_address()


class RayCluster(object):

    def __init__(self, port: int = None, n_workers: int = None, worker_time: int = 3600,
                 verbose: bool = True, ifname='eno1'):
        if port is None:
            port = 44276
        else:
            try:
                int(port)
            except Exception:
                raise TypeError("Invalid port arg: {}".format(port))
        if n_workers is None:
            self.n_workers = 0
        else:
            assert isinstance(n_workers, int)
            self.n_workers = n_workers
        assert isinstance(verbose, bool)
        self.ifname = ifname
        self.verbose = verbose
        self.worker_time = worker_time
        self.port = port
        self.host = get_ip_address(self.ifname)
        self.addr = '{}:{}'.format(self.host, self.port)
        worker_task_config = worker_task.format(self.addr, self.worker_time)
        self.worker_file = 'ray_worker_script'
        with open(self.worker_file, 'w') as f:
            f.write(worker_task_config)

    def _start_ray_head(self):
        if ray.is_initialized():
            raise Exception("Ray is already initialized")
        cmd = 'ray start --head --node-ip-address={} --redis-port={}'.format(self.host, self.port)
        os.system(cmd)

    def _start_ray_workers(self):
        for i in range(self.n_workers):
            os.system('qsub {}'.format(self.worker_file))

    def get_worker_ips(self):
        return set(ray.get([get_workers.remote() for _ in range(1000)]))

    def wait_for_workers(self):
        ips = set([])
        while len(ips) != self.n_workers
            ips = self.get_worker_ips()
            time.sleep(1e-2)

    def init_cluster(self, wait=True):
        self._start_ray_head()
        self._start_ray_workers()
        if wait:
            self.wait_for_workers()

if __name__ == "__main__":
    rc = RayCluster()
    rc.init_cluster()
