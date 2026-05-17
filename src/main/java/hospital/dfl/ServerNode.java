package hospital.dfl;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

public class ServerNode implements Runnable {
    private final NodeParam nodeParam;
    private ServerSocket serverSocket;
    private volatile boolean running = true;

    public ServerNode(NodeParam nodeParam) {
        this.nodeParam = nodeParam;
    }

    @Override
    public void run() {
        try {
            serverSocket = new ServerSocket(nodeParam.getPort());
            while (running) {
                try {
                    Socket clientSocket = serverSocket.accept();
                    new Thread(new ConnectionHandler(clientSocket)).start();
                } catch (IOException e) {
                    if (running) {
                        e.printStackTrace();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void shutdown() {
        running = false;
        if (serverSocket != null && !serverSocket.isClosed()) {
            try {
                serverSocket.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}