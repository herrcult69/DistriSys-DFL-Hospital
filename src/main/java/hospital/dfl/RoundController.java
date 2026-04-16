package hospital.dfl;

import java.util.ArrayList;
import java.util.List;

public class RoundController {
    private final NodeParam params;
    private final ClientNode client;
    private final int[] peerPorts;
    private final String peerHost;
    private final int totalRounds;
    private final int minPeers; // minimum peers needed to do Federated Learning Calculation

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