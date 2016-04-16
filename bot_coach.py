import sys
from collections import deque
sys.path.append('gen-py')


from botcoach import Coach

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

MEMORY_SIZE = 600000
class CoachHandler:
    def __init__(self):
        self.observations = deque()     
        pass

    def botInit(self):
        print 'received bot init'

    def newObservation(self, lastState,lastAction, currentState, reward):
        print 'received observation'
        print ' '.join(str(x) for x in lastState)
        self.observations.append((lastState,lastAction,reward,currentState))
        if(len(self.observations) > MEMORY_SIZE):
            self.observations.popleft()


if __name__ == '__main__':
    handler = CoachHandler()
    processor = Coach.Processor(handler)
    transport = TSocket.TServerSocket(port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    server.serve()