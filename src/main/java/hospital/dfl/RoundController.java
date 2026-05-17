package hospital.dfl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class RoundController {

    private final NodeParam nodeParam;
    private final int totalRounds;
    private final String pythonBase;
    private static final long TRAIN_TIMEOUT_MS     = 60 * 60 * 1000;  // 60 min
    private static final long AGGREGATE_TIMEOUT_MS = 15 * 60 * 1000;  // 15 min


    // Weight-server addresses for all nodes on the LAN.
    // FIX (Bug 2 / Issue 2): ports match fl_server.py formula: WEIGHT_PORT = 5200 + NODE_ID
    // For a single-machine prototype all IPs are 127.0.0.1.
    // For a real LAN deployment, replace IPs with each machine's LAN IP.
    private static final Map<Integer, String> ALL_WEIGHT_ADDRS;
    static {
        ALL_WEIGHT_ADDRS = new HashMap<>();
        ALL_WEIGHT_ADDRS.put(1, "127.0.0.1:5201");
        ALL_WEIGHT_ADDRS.put(2, "127.0.0.1:5202");
        ALL_WEIGHT_ADDRS.put(3, "127.0.0.1:5203");
    }

    public RoundController(NodeParam nodeParam, int totalRounds) {
        this.nodeParam   = nodeParam;
        this.totalRounds = totalRounds;
        // FIX (Bug 2): IPC port = 5100 + nodeId, matching fl_server.py LOCAL_PORT formula
        this.pythonBase  = "http://127.0.0.1:" + (5100 + nodeParam.getNodeId());
    }

    public void runRounds() throws Exception {
        for (int round = 1; round <= totalRounds; round++) {
            System.out.printf("=== Round %d / %d ===%n", round, totalRounds);
            runRound(round);
        }
        System.out.println("Federated Learning complete!");
    }

    private void runRound(int round) throws Exception {
        // Step 1 — Broadcast ROUND_START to peers only (not self)
        broadcastTCP(String.format("{\"type\":\"ROUND_START\",\"round\":%d}", round));

        // Step 2 — Trigger local Python training
        System.out.println("[Java] Triggering local Python training...");
        httpPost(pythonBase + "/train", String.format("{\"round\":%d}", round));

        // Step 3 — Poll until training complete
        pollUntilStatus("training_complete", TRAIN_TIMEOUT_MS);
        System.out.println("[Java] Training complete.");

        // Step 4 — Notify peers this node finished training
        broadcastTCP(String.format("{\"type\":\"ROUND_COMPLETE\",\"node\":%d,\"round\":%d}",
                nodeParam.getNodeId(), round));

        // Step 5 — Wait for ROUND_COMPLETE from all peers
        // FIX (Bug 1): self is pre-added to finishedNodes so we don't need to self-send
        // and don't risk leaking stale ROUND_START messages into the barrier.
        waitForAllPeers("ROUND_COMPLETE", round);
        System.out.println("[Java] All nodes finished Round " + round + ". Starting aggregation.");

        // Step 6 — Trigger local Python aggregation with PEER weight addresses (excluding self)
        // FIX (Issue 2): buildPeersJson filters out self so Python never HTTP-downloads its own adapter
        String peersJson = buildPeersJson(ALL_WEIGHT_ADDRS);
        httpPost(pythonBase + "/aggregate",
                String.format("{\"round\":%d,\"peers\":%s}", round, peersJson));

        // Step 7 — Poll until aggregation complete
        pollUntilStatus("aggregate_complete", AGGREGATE_TIMEOUT_MS);
        System.out.printf("[Java] Node %d Round %d fully complete.%n", nodeParam.getNodeId(), round);
    }

    // ── FIX (Bug 1): Only broadcast to peers — never to self ─────────────────
    private void broadcastTCP(String json) {
        for (int peerPort : nodeParam.getPeerPorts()) {
            ClientNode.sendMessage("127.0.0.1", peerPort, json);
        }
    }

    // ── FIX (Issue 2): Filter out self from peer weight address map ───────────
    private String buildPeersJson(Map<Integer, String> allAddrs) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<Integer, String> entry : allAddrs.entrySet()) {
            if (entry.getKey() == nodeParam.getNodeId()) continue; // skip self
            if (!first) sb.append(",");
            sb.append(String.format("\"%d\":\"%s\"", entry.getKey(), entry.getValue()));
            first = false;
        }
        sb.append("}");
        return sb.toString();
    }

    private void pollUntilStatus(String targetStatus, long timeoutMs) throws InterruptedException {
        long start = System.currentTimeMillis();
        while (System.currentTimeMillis() - start < timeoutMs) {
            try {
                String resp = httpGet(pythonBase + "/status");
                if (resp.contains("\"phase\":\"" + targetStatus + "\"")
                        || resp.contains("\"phase\": \"" + targetStatus + "\"")) {
                    return;
                }
            } catch (Exception ignored) {
                // Python may be mid-startup; keep polling
            }
            Thread.sleep(2000);
        }
        throw new RuntimeException("Timeout waiting for Python status: " + targetStatus);
    }

    // ── FIX (Bug 1): Pre-add self so no self-send is needed ──────────────────
    private void waitForAllPeers(String msgType, int currentRound) throws InterruptedException {
        Set<Integer> finishedNodes = new HashSet<>();
        finishedNodes.add(nodeParam.getNodeId()); // self is already done
        int required = nodeParam.getPeerPorts().size() + 1; // total nodes in cluster

        while (finishedNodes.size() < required) {
            String msg = MessageStore.takeMessage();
            if (msg.contains("\"type\":\"" + msgType + "\"")
                    && msg.contains("\"round\":" + currentRound)) {
                int idx    = msg.indexOf("\"node\":");
                int endIdx = msg.indexOf(",", idx);
                if (endIdx == -1) endIdx = msg.indexOf("}", idx);
                try {
                    int nId = Integer.parseInt(msg.substring(idx + 7, endIdx).trim());
                    finishedNodes.add(nId);
                    System.out.println("[Java] Received " + msgType + " from Node " + nId);
                } catch (NumberFormatException e) {
                    System.err.println("[Java] Could not parse node id from: " + msg);
                }
            } else {
                // Not the message we need — put it back for other consumers
                MessageStore.addMessage(msg);
                Thread.sleep(100);
            }
        }
    }

    // ── HTTP helpers ──────────────────────────────────────────────────────────
    private static String httpPost(String url, String jsonBody) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        conn.setConnectTimeout(5_000);
        conn.setReadTimeout(300_000);
        try (OutputStream os = conn.getOutputStream()) {
            os.write(jsonBody.getBytes(StandardCharsets.UTF_8));
        }
        return readResponse(conn);
    }

    private static String httpGet(String url) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(5_000);
        conn.setReadTimeout(5_000);
        return readResponse(conn);
    }

    private static String readResponse(HttpURLConnection conn) throws IOException {
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = br.readLine()) != null) sb.append(line);
            return sb.toString();
        }
    }
}