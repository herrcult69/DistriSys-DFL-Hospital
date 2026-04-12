import argparse
import numpy as np
import time
import threading
from xmlrpc.server import SimpleXMLRPCServer
from socketserver import ThreadingMixIn
import xmlrpc.client
from http.client import HTTPConnection
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


class TimeoutTransport(xmlrpc.client.Transport):
    def __init__(self, timeout=3.0, use_datetime=False):
        super().__init__(use_datetime=use_datetime)
        self.timeout = timeout

    def make_connection(self, host):
        conn = HTTPConnection(host, timeout=self.timeout)
        return conn


class SimpleLogisticRegression:
    def __init__(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lock = threading.Lock()

    def predict_proba(self, X):
        with self.lock:
            w = self.w.copy()
            b = self.b
        z = np.dot(X, w) + b
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def train_local(self, X, y, epochs=5, lr=0.01):
        for _ in range(epochs):
            with self.lock:
                y_hat = 1 / (1 + np.exp(-(np.dot(X, self.w) + self.b)))
                n = len(X)
                dw = (1 / n) * np.dot(X.T, (y_hat - y))
                db = (1 / n) * np.sum(y_hat - y)
                self.w -= lr * dw
                self.b -= lr * db

    def get_params(self):
        with self.lock:
            return self.w.tolist(), float(self.b)

    def set_params(self, w, b):
        with self.lock:
            self.w = np.array(w, dtype=float)
            self.b = float(b)


def load_local_data(node_id):
    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n = len(X_train)
    splits = [
        (0, int(0.4 * n)),
        (int(0.4 * n), int(0.7 * n)),
        (int(0.7 * n), n)
    ]

    start, end = splits[node_id - 1]

    return X_train[start:end], y_train[start:end], X_test, y_test


class DFLNode:
    def __init__(self, node_id, port, peers):
        self.node_id = node_id
        self.port = port
        self.peers = peers
        self.current_round = 0

        self.X, self.y, self.X_test, self.y_test = load_local_data(node_id)
        self.model = SimpleLogisticRegression(self.X.shape[1])

        self.server = ThreadedXMLRPCServer(("0.0.0.0", port), allow_none=True, logRequests=False)
        self.server.register_function(self.get_status, "get_status")
        self.server.register_function(self.get_weights, "get_weights")

        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        print(f"Node {self.node_id} started on port {self.port}")
        print(f"Local dataset size: {len(self.X)} samples")

    def get_status(self):
        return {
            "node_id": self.node_id,
            "round": self.current_round,
            "samples": len(self.X)
        }

    def get_weights(self):
        w, b = self.model.get_params()
        return {
            "node_id": self.node_id,
            "round": self.current_round,
            "w": w,
            "b": b
        }

    def make_client(self, peer_port, timeout=3.0):
        transport = TimeoutTransport(timeout=timeout)
        return xmlrpc.client.ServerProxy(
            f"http://localhost:{peer_port}",
            transport=transport,
            allow_none=True
        )

    def wait_for_peers(self, retries=10, delay=1.0):
        print("Checking peer availability...")
        for peer in self.peers:
            connected = False
            for _ in range(retries):
                try:
                    client = self.make_client(peer, timeout=2.0)
                    status = client.get_status()
                    print(f"Peer {peer} is up: node {status['node_id']}")
                    connected = True
                    break
                except Exception:
                    time.sleep(delay)

            if not connected:
                print(f"Peer {peer} not reachable at startup. Continuing anyway.")

    def train(self, rounds=10, local_epochs=5, lr=0.01):
        self.wait_for_peers()

        for r in range(1, rounds + 1):
            print(f"\n--- Node {self.node_id} | Round {r} ---")

            self.model.train_local(self.X, self.y, epochs=local_epochs, lr=lr)
            self.current_round = r

            local_w, local_b = self.model.get_params()
            weights = [np.array(local_w)]
            biases = [local_b]

            for peer in self.peers:
                try:
                    client = self.make_client(peer, timeout=2.0)
                    res = client.get_weights()
                    weights.append(np.array(res["w"]))
                    biases.append(res["b"])
                    print(f"Fetched weights from peer {peer} (peer round {res['round']})")
                except Exception as e:
                    print(f"Skipped peer {peer}: {e}")

            avg_w = np.mean(weights, axis=0)
            avg_b = np.mean(biases)

            self.model.set_params(avg_w, avg_b)

            preds = self.model.predict(self.X_test)
            acc = np.mean(preds == self.y_test)

            print(f"Round {r} done | Combined models: {len(weights)} | Test accuracy: {acc:.4f}")

            time.sleep(1)

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Node ID: 1, 2, or 3")
    parser.add_argument("--port", type=int, required=True, help="Port for this node")
    parser.add_argument("--peers", type=int, nargs="*", default=[], help="Ports of peer nodes")
    args = parser.parse_args()

    node = DFLNode(args.id, args.port, args.peers)

    try:
        node.train(rounds=10, local_epochs=5, lr=0.01)
    except KeyboardInterrupt:
        print("\nShutting down node...")
    finally:
        node.shutdown()