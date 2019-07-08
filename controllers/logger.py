import sys

class Logger():
    def __init__(self, cc):
        self.rank = cc["mpi"]["rank"]

    def to_log(self, message):
        print("{}) ".format(self.rank) + message)
        sys.stdout.flush()