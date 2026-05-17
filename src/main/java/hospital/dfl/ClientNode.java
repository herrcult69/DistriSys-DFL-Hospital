package hospital.dfl;

import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

public class ClientNode {

    public static void sendMessage(String ip, int port, String json) {
        try (Socket socket = new Socket(ip, port)) {
            byte[] data = json.getBytes(StandardCharsets.UTF_8);
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            out.writeInt(data.length);
            out.write(data);
            out.flush();
        } catch (IOException e) {
            System.err.println("Failed to send message to " + ip + ":" + port + " - " + e.getMessage());
        }
    }
}