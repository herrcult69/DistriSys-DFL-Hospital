package hospital.dfl;

// Command to run code:
//
//  *   mvn exec:java -Dexec.mainClass="hospital.dfl.MainNode" -Dexec.args="1 8001 8002 8003"
//  *   mvn exec:java -Dexec.mainClass="hospital.dfl.MainNode" -Dexec.args="2 8002 8001 8003"
//  *   mvn exec:java -Dexec.mainClass="hospital.dfl.MainNode" -Dexec.args="3 8003 8001 8002"

public class MainNode {

    private static final int NUM_FEATURES  = 4;  // placeholder
    private static final int TOTAL_ROUNDS  = 1;

    public static void main(String[] args) throws InterruptedException {
        if (args.length < 2) {
            System.err.println("Please use: MainNode <nodeId> <port> [peerPorts...]");
            System.exit(1);
        }
        
        int nodeId = Integer.parseInt(args[0]);
        int port   = Integer.parseInt(args[1]);
        
        // Parse peer ports from remaining args
        int[] peerPorts = new int[args.length - 2];
        for (int i = 0; i < peerPorts.length; i++) {
            peerPorts[i] = Integer.parseInt(args[i + 2]);
        }
        
        
        // current parameters
        NodeParam currentParams = new NodeParam(nodeId, port, NUM_FEATURES);

        // server runs on a separate thread so the main flow can coordinate rounds
        ServerNode server = new ServerNode(port, currentParams);
        Thread serverThread = new Thread(server);
        serverThread.start();

        try {
            // Give server time to bind before sending requests
            Thread.sleep(2000); // wait for user to run three terminal

            // Coordinator for each round
            RoundController controller = new RoundController(currentParams, peerPorts, TOTAL_ROUNDS);
            controller.runRounds();
        } finally {
            server.shutdown(); // shut connection down on main thread causing connection to close waking up blocked accept()
            serverThread.join();
        }
    }
}