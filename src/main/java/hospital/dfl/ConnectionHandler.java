package hospital.dfl;

import java.net.*;
import java.io.*;

public class ConnectionHandler implements Runnable {
    private final Socket socket;
    private final NodeParam params;

    public ConnectionHandler(Socket socket, NodeParam params) {
        this.socket = socket;
        this.params = params;
    }

    public void run() {
        try (
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true)
        ) {
            String request = in.readLine();
            if (request == null) return;
            System.out.println("[Node " + params.getNodeId() + "] Received: " + request);
            String response = params.jsonifyMsg();
            out.println(response);
        } catch (Exception e) {
            System.err.println("ConnectionHandler Error: " + e.getMessage());
        } finally {
            try { 
                socket.close(); 
            } catch (IOException ignored) {
            }
        }
    }
}