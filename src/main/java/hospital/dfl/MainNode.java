package hospital.dfl;

import java.util.ArrayList;
import java.util.List;

public class MainNode {
    private static final int TOTAL_ROUNDS = 2; // For quick prototype demo

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: MainNode <nodeId> <port> [peerPorts...]");
            System.exit(1);
        }

        int nodeId = Integer.parseInt(args[0]);
        int port = Integer.parseInt(args[1]);

        List<Integer> peerPorts = new ArrayList<>();
        for (int i = 2; i < args.length; i++) {
            peerPorts.add(Integer.parseInt(args[i]));
        }

        NodeParam currentParams = new NodeParam(nodeId, port, peerPorts);

        ServerNode server = new ServerNode(currentParams);
        Thread serverThread = new Thread(server);
        serverThread.start();

        System.out.println("Started Java Coordinator Node " + nodeId + " on port " + port);
        System.out.println("Waiting 5 seconds for peers to start...");
        Thread.sleep(5000);

        try {
            RoundController controller = new RoundController(currentParams, TOTAL_ROUNDS);
            controller.runRounds();
        } finally {
            server.shutdown();
            serverThread.join();
        }
    }
}
