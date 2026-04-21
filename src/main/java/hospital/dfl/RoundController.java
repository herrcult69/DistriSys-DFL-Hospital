package hospital.dfl;

import java.util.ArrayList;
import java.util.List;

import java.io.DataOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class RoundController {
    private final NodeParam params;
    private final ClientNode client;
    private final int[] peerPorts;
    private final String peerHost;
    private final int totalRounds;
    private final int minPeers; // minimum peers needed to do Federated Learning Calculation

    private static final byte MSG_AGGREGATED_PARAMS = 2;
    private static final byte VERSION = 1;
    private static final byte[] MAGIC = new byte[]{'D', 'F', 'L', '1'};

    private void sendParamsToPython(DataOutputStream out, int round, double[] weights, double bias) throws Exception{
        ByteBuffer payload = ByteBuffer.allocate(4 + (weights.length * 4) + 4).order(ByteOrder.LITTLE_ENDIAN);
        payload.putInt(weights.length);
        for (double w : weights){
            payload.putFloat((float) w);
        }
        payload.putFloat((float) bias);

        byte[] p = payload.array();
        out.write(MAGIC);
        out.writeByte(VERSION);
        out.writeByte(MSG_AGGREGATED_PARAMS);
        out.writeInt(round);
        out.writeInt(p.length);
        out.write(p);
        out.flush();
    }

    public RoundController(NodeParam params, int[] peerPorts, int totalRounds) {
        this.params = params;
        this.client = new ClientNode(params.getNodeId());
        this.peerPorts = peerPorts;
        this.peerHost = "localhost";
        this.totalRounds = totalRounds;
        this.minPeers = 1;
    }

    public void runRounds() throws InterruptedException  {
        for (int round = 0; round < totalRounds; round++) {
            System.out.println("\nNode " + params.getNodeId() + ": Round " + round + "...");
            params.setCurrentRound(round);

            // Step 1: local training happens here (Python bridge later)
            // 
            System.out.println("Python Script running, training locally...");
            Thread.sleep(1000);
            // param.setWeights(x[], x)
            System.out.println("Trainning complete, nothing changed");

            // Step 2: request weights from all peers
            List<String> responses = new ArrayList<>();
            for (int peerPort : peerPorts) {
                String response = client.requestWeights(peerHost, peerPort, round);
                if (response != null) {
                    System.out.println("Node " + params.getNodeId() + ", Got from port " + peerPort + ": " + response);
                    responses.add(response);
                }
            }

            // Step 3: check threshold
            if (responses.size() >= minPeers) {
                System.out.println("Enough peers data, doing aggregation algorithm");
                // TODO: parse JSON like responses and call caculate new weights here
            } else {
                System.out.println("No peers responded — using own weights only");
            }

            // Step 4: small pause between rounds
            
            Thread.sleep(1000);
           
        }
        System.out.println("[Node " + params.getNodeId() + "] All rounds complete.");
    }
}