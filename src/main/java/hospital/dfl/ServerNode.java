package hospital.dfl;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;

public class ServerNode implements Runnable {
    private final int port;
    private final NodeParam params;
    private volatile boolean running = true;
    private volatile ServerSocket serverSocket;

    public ServerNode(int port, NodeParam params) {
        this.port = port;
        this.params = params;
    }

    public void shutdown() {
        running = false;
        if (serverSocket != null && !serverSocket.isClosed()) {
            try {
                serverSocket.close();
            } catch (IOException ignored) {
            }
        }
    }

    @Override
    public void run() {
        try {
            serverSocket = new ServerSocket(port);
            System.out.println("Node " + params.getNodeId() + ": Java Server listening on port " + port);
            while (running) {
                try {
                    Socket client = serverSocket.accept();
                    Thread t = new Thread(new ConnectionHandler(client, params));
                    t.start();
                } catch (SocketException e) {
                    if (running) {
                        throw e;
                    }
                    break;
                }
            }
        } catch (Exception e) {
            if (running) {
                e.printStackTrace();
            }
        } 
    }

}
