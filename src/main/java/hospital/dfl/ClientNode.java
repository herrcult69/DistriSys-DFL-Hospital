package hospital.dfl;
import java.net.*;
import java.io.*;

public class ClientNode {
    private final int nodeId;

    public ClientNode(int nodeId) {
        this.nodeId = nodeId;
    }
    
    public String requestWeights(String peerAddr, int peerPort, int round) {
        try (Socket socket = new Socket(peerAddr, peerPort)) {
            socket.setSoTimeout(3000);
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));

            // send dummy request to peers
            out.println("{\"type\":\"getparam\"" + 
                        ",\"from\":" + this.nodeId + 
                        ",\"round\":" + round + "}");

            return in.readLine().trim();
        } catch (SocketTimeoutException e) {
            System.err.println("Peer:" + peerPort + " Timed out");
            return null;
        } catch (IOException e) {
            System.err.println("Peer:" + peerPort + " Unreachable");
            return null;
        }
    }
}
