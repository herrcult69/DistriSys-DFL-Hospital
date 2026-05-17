package hospital.dfl;

import java.util.List;

public class NodeParam {
    private final int nodeId;
    private final int port;
    private final List<Integer> peerPorts;

    public NodeParam(int nodeId, int port, List<Integer> peerPorts) {
        this.nodeId = nodeId;
        this.port = port;
        this.peerPorts = peerPorts;
    }

    public int getNodeId() {
        return nodeId;
    }

    public int getPort() {
        return port;
    }

    public List<Integer> getPeerPorts() {
        return peerPorts;
    }
}