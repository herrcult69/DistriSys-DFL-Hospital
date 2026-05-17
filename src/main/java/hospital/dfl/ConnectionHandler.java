package hospital.dfl;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

public class ConnectionHandler implements Runnable {
    private final Socket socket;

    public ConnectionHandler(Socket socket) {
        this.socket = socket;
    }

    @Override
    public void run() {
        try (socket) {
            DataInputStream in = new DataInputStream(socket.getInputStream());
            int length = in.readInt();
            byte[] data = new byte[length];
            in.readFully(data);
            String message = new String(data, StandardCharsets.UTF_8);
            
            System.out.println("[Received] " + message);
            MessageStore.addMessage(message);

        } catch (IOException e) {
            System.out.println("Connection handler error: " + e.getMessage());
        }
    }
}