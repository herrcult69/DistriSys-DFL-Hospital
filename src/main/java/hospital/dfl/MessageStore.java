package hospital.dfl;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class MessageStore {
    private static final BlockingQueue<String> messages = new LinkedBlockingQueue<>();

    public static void addMessage(String msg) {
        messages.offer(msg);
    }

    public static String takeMessage() throws InterruptedException {
        return messages.take();
    }
}