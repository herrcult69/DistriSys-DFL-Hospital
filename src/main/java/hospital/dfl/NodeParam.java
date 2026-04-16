package hospital.dfl;

public class NodeParam {
    private final int nodeId;
    private final int port;
    private double[] weights;
    private double bias;
    private int currentRound;

    public NodeParam(int nodeId, int port, int numFeatures) {
        this.nodeId = nodeId;
        this.port = port;
        this.weights = new double[numFeatures];
        this.bias = 0.0;
        this.currentRound = 0;

    }
       
    public int getNodeId() { return nodeId; }
    public int getPort() { return port; }

   
    public synchronized int getCurrentRound() { return currentRound; }
    public synchronized void setCurrentRound(int r) { this.currentRound = r; }

    public synchronized double[] getWeights() { return weights.clone(); }
    public synchronized double getBias() { return bias; }
    public synchronized void setWeights(double[] w, double b) {
        this.weights = w.clone();
        this.bias = b;
    }
    public synchronized String jsonifyMsg() {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"nodeId\":").append(this.nodeId).append(",");
        sb.append("\"round\":").append(this.currentRound).append(",");
        sb.append("\"bias\":").append(this.bias).append(",");
        sb.append("\"weights\":[");
        for (int i = 0; i < this.weights.length; i++) {
            sb.append(this.weights[i]);
            if (i < this.weights.length - 1) sb.append(",");
        }
        sb.append("]}");
        return sb.toString();
    }
}