package hospital.dfl;

import java.util.ArrayList;
import java.util.List;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class RoundController {
    private final NodeParam params;
    private final ClientNode client;
    private final int[] peerPorts;
    private final String peerHost;
    private final int totalRounds;
    private final int minPeers; // minimum peers needed to do Federated Learning Calculation

    private static final byte MSG_TRAIN_RESULT = 1;
    private static final byte MSG_AGGREGATED_PARAMS = 2;
    private static final byte MSG_SHUTDOWN = 3;

    private static final byte VERSION = 1;
    private static final byte[] MAGIC = new byte[]{'D', 'F', 'L', '1'};

    private static final String PY_HOST = "127.0.0.1";
    private static final int PY_BASE_PORT = 5000;
    private static final int PY_TIMEOUT_MS = 120_000;

    public RoundController(NodeParam params, int[] peerPorts, int totalRounds) {
        this.params = params;
        this.client = new ClientNode(params.getNodeId());
        this.peerPorts = peerPorts;
        this.peerHost = "localhost";
        this.totalRounds = totalRounds;
        this.minPeers = 1;
    }

    public void runRounds() throws InterruptedException  {
        int pyPort = PY_BASE_PORT + params.getNodeId();
        try(
            Socket pySocket = new Socket(PY_HOST, pyPort);
            DataInputStream in = new DataInputStream(new BufferedInputStream(pySocket.getInputStream()));
            DataOutputStream out = new DataOutputStream(new BufferedOutputStream(pySocket.getOutputStream()));

        ){
            pySocket.setSoTimeout(PY_TIMEOUT_MS);
            System.out.println("Connected to Python at " + PY_HOST + ":"  + pyPort);

            for (int round = 1; round <= totalRounds; round++){
                params.setCurrentRound(round);
                System.out.println("\nNode " + params.getNodeId() + " : Round " + round);

                // 1) Java waits while Python trains
                Frame trainResult = readFrame(in);
                if (trainResult.type != MSG_TRAIN_RESULT){
                    throw new IllegalStateException("Expected TRAIN_RESULT, got type=" + trainResult.type);

                }

                // 2) Update NodeParam from Python result
                params.setWeights(trainResult.weights, trainResult.bias);
                System.out.println("Received params from Python after round " + trainResult.round);

                // 3) Connection test only: send back same params directly
                sendParamsToPython(out,MSG_AGGREGATED_PARAMS, round, params.getWeights(), params.getBias());
                out.flush();
                System.out.println("Sent params back to Python for round " + (round+1));

            }   
            sendParamsToPython(out, MSG_SHUTDOWN, totalRounds, params.getWeights(), params.getBias());
            out.flush();

        } catch (SocketTimeoutException e){
            System.err.println("Python socket timeout: " + e.getMessage());
        } catch (Exception e){
            System.err.println("RoundController Error: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("[Node " + params.getNodeId() + "] All rounds complete.");

        // for (int round = 0; round < totalRounds; round++) {
        //     System.out.println("\nNode " + params.getNodeId() + ": Round " + round + "...");
        //     params.setCurrentRound(round);

        //     // Step 1: local training happens here (Python bridge later)
        //     // 
        //     System.out.println("Python Script running, training locally...");
        //     Thread.sleep(1000);
        //     // param.setWeights(x[], x)
        //     System.out.println("Trainning complete, nothing changed");

        //     // Step 2: request weights from all peers
        //     List<String> responses = new ArrayList<>();
        //     for (int peerPort : peerPorts) {
        //         String response = client.requestWeights(peerHost, peerPort, round);
        //         if (response != null) {
        //             System.out.println("Node " + params.getNodeId() + ", Got from port " + peerPort + ": " + response);
        //             responses.add(response);
        //         }
        //     }

        //     // Step 3: check threshold
        //     if (responses.size() >= minPeers) {
        //         System.out.println("Enough peers data, doing aggregation algorithm");
        //         // TODO: parse JSON like responses and call caculate new weights here
        //     } else {
        //         System.out.println("No peers responded — using own weights only");
        //     }

        //     // Step 4: small pause between rounds
            
        //     Thread.sleep(1000);
           
        // }
        // System.out.println("[Node " + params.getNodeId() + "] All rounds complete.");
    }

    private void sendParamsToPython(DataOutputStream out, byte msgType, int round, double[] weights, double bias) throws Exception{
        ByteBuffer payload = ByteBuffer.allocate(4 + (weights.length * 4) + 4).order(ByteOrder.LITTLE_ENDIAN);

        payload.putInt(weights.length);
        for (double w : weights){
            payload.putFloat((float) w);
        }

        payload.putFloat((float) bias);

        byte[] p = payload.array();
        out.write(MAGIC);
        out.writeByte(VERSION);
        out.writeByte(msgType);
        out.writeInt(round);
        out.writeInt(p.length);
        out.write(p);
        out.flush();
    }

    private Frame readFrame(DataInputStream in) throws Exception{
        byte[] magic = new byte[4];
        in.readFully(magic);
        for (int i = 0; i < 4; i++){
            if (magic[i] != MAGIC[i]){
                throw new IllegalStateException("Bad magic header");
            }
        }

        byte version = in.readByte();
        if (version != VERSION){
            throw new IllegalStateException("Unsupported version: " + version);
        }

        byte type = in.readByte();
        int round = in.readInt();
        int payloadLen = in.readInt();
        if (payloadLen <= 0 || payloadLen > 32 * 1024 * 1024){
            throw new IllegalStateException("Invalid payload length: " + payloadLen);
        }

        byte[] payload = new byte[payloadLen];
        in.readFully(payload);

        ByteBuffer bb = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        int n = bb.getInt();
        if (n <= 0 || n > 1_000_000_000){
            throw new IllegalStateException("Invalid weight count: " + n);
        }
        double[] weights = new double[n];
        for (int i = 0; i < n; i++){
            weights[i] = bb.getFloat();
        }

        double bias = bb.getFloat();

        return new Frame(type, round, weights, bias);
    }

    private static class Frame{
        final byte type;
        final int round;
        final double[] weights;
        final double bias;

        Frame(byte type, int round, double[] weights, double bias){
            this.type = type;
            this.round = round;
            this.weights = weights;
            this.bias = bias;
        }
    }
}