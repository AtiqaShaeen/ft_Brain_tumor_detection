"""
FL Client - Sends Training Request to Server
Place in: client1/client1.py or client2/client2.py
"""

import flwr as fl
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import sys
import time
import os
import json
import socket
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------------
# CLIENT CONFIG
# -------------------------------
CLIENT_ID = "client1"  # Change to "client2" for client2
DATASET_PATH = "Braintumors_client1"  # Change accordingly
IMG_SIZE = 224
BATCH_SIZE = 32
LOCAL_EPOCHS = 2
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 3

VERSION_LOG_FILE = "client_version_log.json"
SAVE_DURING_TRAINING = False  # Don't save during rounds
UPDATE_NOTIFICATION_FILE = "model_update_notification.json"


# -------------------------------
# Model Update Notification Manager
# -------------------------------
class ModelUpdateNotification:
    """Manages notifications for available model updates."""
    
    def __init__(self, notification_file: str = UPDATE_NOTIFICATION_FILE):
        self.notification_file = notification_file
        self.notifications = self._load_notifications()
    
    def _load_notifications(self):
        """Load pending notifications."""
        if os.path.exists(self.notification_file):
            try:
                with open(self.notification_file, 'r') as f:
                    return json.load(f)
            except:
                return {"pending_updates": []}
        return {"pending_updates": []}
    
    def _save_notifications(self):
        """Save notifications to file."""
        try:
            with open(self.notification_file, 'w') as f:
                json.dump(self.notifications, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save notification: {e}")
    
    def add_update_notification(self, total_rounds: int, final_accuracy: float, final_loss: float):
        """Add a new model update notification."""
        notification = {
            "timestamp": datetime.now().isoformat(),
            "total_rounds": total_rounds,
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "status": "pending",
            "model_file": f"{CLIENT_ID}_FinalGlobal_v{len(self.notifications['pending_updates']) + 1}.h5"
        }
        self.notifications["pending_updates"].append(notification)
        self._save_notifications()
        return notification
    
    def get_pending_updates(self):
        """Get all pending updates."""
        return [n for n in self.notifications["pending_updates"] if n["status"] == "pending"]
    
    def mark_update_accepted(self, index: int):
        """Mark an update as accepted."""
        if 0 <= index < len(self.notifications["pending_updates"]):
            self.notifications["pending_updates"][index]["status"] = "accepted"
            self.notifications["pending_updates"][index]["accepted_at"] = datetime.now().isoformat()
            self._save_notifications()
    
    def mark_update_rejected(self, index: int):
        """Mark an update as rejected."""
        if 0 <= index < len(self.notifications["pending_updates"]):
            self.notifications["pending_updates"][index]["status"] = "rejected"
            self.notifications["pending_updates"][index]["rejected_at"] = datetime.now().isoformat()
            self._save_notifications()
    
    def print_pending_updates(self):
        """Display pending updates."""
        pending = self.get_pending_updates()
        
        if not pending:
            print(f"\n✅ No pending model updates\n")
            return
        
        print(f"\n{'='*70}")
        print(f"🔔 MODEL UPDATE AVAILABLE!")
        print(f"{'='*70}")
        
        for i, update in enumerate(pending):
            print(f"\nUpdate #{i + 1}:")
            print(f"   Timestamp: {datetime.fromisoformat(update['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Training Rounds: {update['total_rounds']}")
            print(f"   Final Accuracy: {update['final_accuracy']:.4f}")
            print(f"   Final Loss: {update['final_loss']:.4f}")
            print(f"   Model File: {update['model_file']}")
            print(f"   Status: {update['status'].upper()}")
        
        print(f"{'='*70}\n")


# -------------------------------
# Request Sender
# -------------------------------
class TrainingRequestSender:
    """Sends training requests to server."""
    
    @staticmethod
    def send_request(server_host: str, server_port: int, client_id: str, num_rounds: int):
        """Send training request to server."""
        try:
            # Connect to server's request listener
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(10)
            client_socket.connect((server_host, server_port))
            
            # Prepare request
            request_data = {
                'client_id': client_id,
                'num_rounds': num_rounds,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send request
            client_socket.send(json.dumps(request_data).encode('utf-8'))
            
            # Receive response
            response_data = client_socket.recv(4096).decode('utf-8')
            response = json.loads(response_data)
            
            client_socket.close()
            
            return response
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def check_approval_status(server_host: str, fl_port: int, max_attempts: int = 60):
        """
        Check if training request was approved by trying to connect to FL port.
        Returns (approved, port) tuple.
        """
        print(f"\n⏳ Waiting for admin approval...")
        print(f"   Checking if port {fl_port} is ready...")
        
        for attempt in range(max_attempts):
            try:
                # Try to connect to FL server port
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(1)
                result = test_socket.connect_ex((server_host, fl_port))
                test_socket.close()
                
                if result == 0:
                    print(f"✅ Request APPROVED! Port {fl_port} is ready.")
                    return True, fl_port
                
                # Show progress
                if attempt % 10 == 0 and attempt > 0:
                    print(f"   Still waiting... ({attempt}/{max_attempts} checks)")
                
                time.sleep(1)
                
            except Exception:
                time.sleep(1)
        
        print(f"❌ Timeout: No approval received after {max_attempts} seconds")
        return False, None


# -------------------------------
# Version Tracker
# -------------------------------
class ClientVersionTracker:
    """Track model versions."""
    
    def __init__(self, log_file: str = VERSION_LOG_FILE):
        self.log_file = log_file
        self.log = self._load_log()
    
    def _load_log(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except:
                return {"client_id": CLIENT_ID, "versions": []}
        return {"client_id": CLIENT_ID, "versions": []}
    
    def _save_log(self):
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.log, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save log: {e}")
    
    def log_version(self, round_num: int, metrics: dict, model_path: str = None):
        version_entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "model_path": model_path
        }
        self.log["versions"].append(version_entry)
        self._save_log()
    
    def get_last_version(self):
        if self.log["versions"]:
            return self.log["versions"][-1]
        return None
    
    def print_version_history(self):
        if not self.log["versions"]:
            print(f"\n[{CLIENT_ID}] No version history\n")
            return
        
        print(f"\n{'='*70}")
        print(f"[{CLIENT_ID}] 📜 VERSION HISTORY")
        print(f"{'='*70}")
        print(f"{'Round':<8} {'Time':<12} {'Accuracy':<12} {'Loss':<12}")
        print(f"{'-'*70}")
        
        for v in self.log["versions"]:
            round_num = v["round"]
            timestamp = datetime.fromisoformat(v["timestamp"]).strftime("%H:%M:%S")
            acc = v["metrics"].get("accuracy", "N/A")
            loss = v["metrics"].get("loss", "N/A")
            
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else acc
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else loss
            
            print(f"{round_num:<8} {timestamp:<12} {acc_str:<12} {loss_str:<12}")
        
        print(f"{'='*70}\n")


# -------------------------------
# Data Loading
# -------------------------------
def load_data():
    """Load data."""
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        validation_split=0.20
    )
    
    train_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )
    
    val_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )
    
    return train_data, val_data


# -------------------------------
# Model Creation
# -------------------------------
def create_model(num_classes):
    """Create model."""
    base_model = ResNet50(
        weights="imagenet", 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer=Adam(1e-4), 
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )
    
    return model


# -------------------------------
# FL Client
# -------------------------------
class VersionControlledClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, val_data, version_tracker):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.train_samples = train_data.samples
        self.val_samples = val_data.samples
        self.round_count = 0
        self.total_rounds = 0
        self.version_tracker = version_tracker
        self.final_accuracy = 0.0
        self.final_loss = 0.0

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        try:
            self.round_count += 1
            start_time = time.time()
            
            print(f"\n{'='*70}")
            print(f"[{CLIENT_ID}] 🔄 ROUND {self.round_count}")
            print(f"{'='*70}")
            
            # Update model with received parameters
            self.model.set_weights(parameters)
            print(f"[{CLIENT_ID}] 📥 Received global model")
            
            # Train locally (NO SAVING DURING TRAINING)
            print(f"[{CLIENT_ID}] 🧠 Training {LOCAL_EPOCHS} epochs on {self.train_samples} samples")
            
            history = self.model.fit(
                self.train_data, 
                epochs=LOCAL_EPOCHS, 
                verbose=1
            )
            
            accuracy = float(history.history["accuracy"][-1])
            loss = float(history.history["loss"][-1])
            elapsed = time.time() - start_time
            
            # Store final metrics (but don't save model yet)
            self.final_accuracy = accuracy
            self.final_loss = loss
            
            # Log version (without saving model file)
            self.version_tracker.log_version(
                round_num=self.round_count,
                metrics={"accuracy": accuracy, "loss": loss},
                model_path=None  # No model saved during training
            )
            
            print(f"\n[{CLIENT_ID}] ✅ Completed in {elapsed/60:.1f} min")
            print(f"[{CLIENT_ID}] 📊 Acc: {accuracy:.4f}, Loss: {loss:.4f}")
            print(f"[{CLIENT_ID}] 📤 Sending updates...")
            print(f"{'='*70}\n")
            
            return self.model.get_weights(), self.train_samples, {
                "accuracy": accuracy,
                "loss": loss
            }
            
        except Exception as e:
            print(f"[{CLIENT_ID}] ❌ Error: {e}")
            raise

    def evaluate(self, parameters, config):
        try:
            print(f"\n[{CLIENT_ID}] 📊 EVALUATION (Round {self.round_count})")
            
            self.model.set_weights(parameters)
            loss, accuracy = self.model.evaluate(self.val_data, verbose=0)
            
            # Store final metrics
            self.final_accuracy = float(accuracy)
            self.final_loss = float(loss)
            
            print(f"[{CLIENT_ID}] ✅ Loss: {loss:.4f}, Acc: {accuracy:.4f}\n")
            
            return loss, self.val_samples, {"accuracy": float(accuracy)}
            
        except Exception as e:
            print(f"[{CLIENT_ID}] ❌ Error: {e}")
            raise


# -------------------------------
# Client Menu
# -------------------------------
def show_client_menu(version_tracker, update_notifier):
    """Client menu."""
    
    # Check for pending updates
    pending_updates = len(update_notifier.get_pending_updates())
    
    print(f"\n{'='*70}")
    print(f"[{CLIENT_ID}] 🎛️  CLIENT MENU")
    print(f"{'='*70}")
    
    if pending_updates > 0:
        print(f"🔔 {pending_updates} MODEL UPDATE(S) AVAILABLE!")
        print(f"{'='*70}")
    
    print("1. 🚀 Send Training Request to Server")
    print("2. 📜 View Version History")
    print("3. 🔍 View Last Version")
    print("4. 📂 List Local Models")
    print("5. 🔔 View Model Update Notifications")
    print("6. ✅ Accept Model Update")
    print("7. ❌ Reject Model Update")
    print("8. ⚙️  Settings")
    print("9. 🚪 Exit")
    print(f"{'='*70}")
    
    choice = input("\nChoice (1-9): ").strip()
    return choice
    return choice


# -------------------------------
# Main
# -------------------------------
def main():
    global CLIENT_ID, LOCAL_EPOCHS, SAVE_DURING_TRAINING
    
    print(f"\n{'='*70}")
    print(f"🚀 FEDERATED LEARNING CLIENT")
    print(f"{'='*70}")
    print(f"📋 Client ID: {CLIENT_ID}")
    print(f"📂 Dataset: {DATASET_PATH}")
    print(f"{'='*70}")
    
    version_tracker = ClientVersionTracker()
    update_notifier = ModelUpdateNotification()
    
    # Check for pending updates at startup
    pending = update_notifier.get_pending_updates()
    if pending:
        print(f"\n🔔 ATTENTION: You have {len(pending)} pending model update(s)!")
        print(f"   Select option 5 from menu to view details\n")
    
    # Menu loop
    while True:
        choice = show_client_menu(version_tracker, update_notifier)
        
        if choice == "1":
            # Send training request
            print(f"\n{'='*70}")
            print(f"[{CLIENT_ID}] 📤 SEND TRAINING REQUEST")
            print(f"{'='*70}")
            
            server_host = input("Server host (default: 127.0.0.1): ").strip() or "127.0.0.1"
            request_port = input("Request port (default: 9090): ").strip() or "9090"
            request_port = int(request_port)
            
            num_rounds = input("Number of rounds (default: 10): ").strip()
            num_rounds = int(num_rounds) if num_rounds.isdigit() else 10
            
            fl_port = input("Expected FL port (default: 8080): ").strip() or "8080"
            fl_port = int(fl_port)
            
            print(f"\n[{CLIENT_ID}] 📡 Sending request to {server_host}:{request_port}...")
            
            # Send request
            response = TrainingRequestSender.send_request(
                server_host,
                request_port,
                CLIENT_ID,
                num_rounds
            )
            
            if response['status'] == 'received':
                print(f"[{CLIENT_ID}] ✅ Request sent successfully!")
                print(f"[{CLIENT_ID}] 📋 Request ID: {response['request_id']}")
                print(f"[{CLIENT_ID}] 💬 {response['message']}")
                print(f"\n⚠️  IMPORTANT:")
                print(f"   1. Admin must approve your request on the server")
                print(f"   2. Admin should select your Request ID: {response['request_id']}")
                print(f"   3. Once approved, training will start automatically")
                print(f"   4. Models will NOT be saved during training")
                print(f"   5. After training completes, you'll receive a model update notification")
                
                # Wait for approval
                approved, assigned_port = TrainingRequestSender.check_approval_status(
                    server_host,
                    fl_port,
                    max_attempts=120
                )
                
                if approved:
                    server_address = f"{server_host}:{assigned_port}"
                    # Store num_rounds for later
                    total_rounds_requested = num_rounds
                    break  # Exit menu to start training
                else:
                    print(f"\n[{CLIENT_ID}] ❌ Request was not approved or timed out")
                    print(f"[{CLIENT_ID}] Please try again or contact admin\n")
                    continue
            else:
                print(f"\n[{CLIENT_ID}] ❌ Failed to send request")
                print(f"[{CLIENT_ID}] Error: {response.get('message', 'Unknown error')}")
                print(f"[{CLIENT_ID}] Make sure server is running!\n")
                continue
                
        elif choice == "2":
            version_tracker.print_version_history()
            
        elif choice == "3":
            last = version_tracker.get_last_version()
            if last:
                print(f"\n{'='*70}")
                print(f"[{CLIENT_ID}] 📋 LAST VERSION")
                print(f"{'='*70}")
                print(f"   Round: {last['round']}")
                print(f"   Time: {last['timestamp']}")
                print(f"   Metrics: {last['metrics']}")
                if last.get('model_path'):
                    print(f"   Model: {last['model_path']}")
                print(f"{'='*70}\n")
            else:
                print(f"\n[{CLIENT_ID}] No versions yet\n")
                
        elif choice == "4":
            model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
            if model_files:
                print(f"\n{'='*70}")
                print(f"[{CLIENT_ID}] 📂 LOCAL MODELS")
                print(f"{'='*70}")
                for i, f in enumerate(model_files, 1):
                    size = os.path.getsize(f) / (1024 * 1024)
                    mtime = datetime.fromtimestamp(os.path.getmtime(f))
                    print(f"   {i}. {f}")
                    print(f"      Size: {size:.2f} MB")
                    print(f"      Date: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*70}\n")
            else:
                print(f"\n[{CLIENT_ID}] No models found\n")
        
        elif choice == "5":
            # View pending updates
            update_notifier.print_pending_updates()
        
        elif choice == "6":
            # Accept model update
            pending = update_notifier.get_pending_updates()
            if not pending:
                print(f"\n[{CLIENT_ID}] ✅ No pending updates to accept\n")
                continue
            
            update_notifier.print_pending_updates()
            idx = input("\nEnter update number to accept (or 0 to cancel): ").strip()
            
            if idx.isdigit() and int(idx) > 0:
                idx = int(idx) - 1
                if 0 <= idx < len(pending):
                    update_notifier.mark_update_accepted(idx)
                    print(f"\n✅ Update accepted!")
                    print(f"   Model file: {pending[idx]['model_file']}")
                    print(f"   You can now use this model for inference/deployment\n")
                else:
                    print(f"\n❌ Invalid update number\n")
        
        elif choice == "7":
            # Reject model update
            pending = update_notifier.get_pending_updates()
            if not pending:
                print(f"\n[{CLIENT_ID}] ✅ No pending updates to reject\n")
                continue
            
            update_notifier.print_pending_updates()
            idx = input("\nEnter update number to reject (or 0 to cancel): ").strip()
            
            if idx.isdigit() and int(idx) > 0:
                idx = int(idx) - 1
                if 0 <= idx < len(pending):
                    confirm = input(f"   Are you sure you want to reject this update? (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        update_notifier.mark_update_rejected(idx)
                        print(f"\n❌ Update rejected\n")
                    else:
                        print(f"\n✅ Rejection cancelled\n")
                else:
                    print(f"\n❌ Invalid update number\n")
                    
        elif choice == "8":
            print(f"\n{'='*70}")
            print(f"[{CLIENT_ID}] ⚙️  SETTINGS")
            print(f"{'='*70}")
            
            new_id = input(f"Client ID (current: {CLIENT_ID}): ").strip()
            if new_id:
                CLIENT_ID = new_id
                version_tracker = ClientVersionTracker()
                update_notifier = ModelUpdateNotification()
            
            new_epochs = input(f"Local epochs (current: {LOCAL_EPOCHS}): ").strip()
            if new_epochs.isdigit():
                LOCAL_EPOCHS = int(new_epochs)
            
            print(f"\n✅ Settings updated!\n")
            
        elif choice == "9":
            print(f"\n[{CLIENT_ID}] 👋 Goodbye!\n")
            sys.exit(0)
            
        else:
            print(f"\n[{CLIENT_ID}] ❌ Invalid choice\n")
    
    # Load data
    print(f"\n[{CLIENT_ID}] 📂 Loading dataset...")
    try:
        train_data, val_data = load_data()
    except Exception as e:
        print(f"[{CLIENT_ID}] ❌ Error loading data: {e}")
        sys.exit(1)
    
    num_classes = train_data.num_classes
    
    print(f"\n[{CLIENT_ID}] 📊 Dataset:")
    print(f"   Classes: {list(train_data.class_indices.keys())}")
    print(f"   Training: {train_data.samples}")
    print(f"   Validation: {val_data.samples}")
    
    # Create model
    print(f"\n[{CLIENT_ID}] 🧠 Building model...")
    model = create_model(num_classes)
    print(f"[{CLIENT_ID}] ✅ Model ready")
    
    # Create client
    client = VersionControlledClient(model, train_data, val_data, version_tracker)
    client.total_rounds = total_rounds_requested
    
    # Connect to server
    print(f"\n[{CLIENT_ID}] 🌐 Connecting to {server_address}...")
    print(f"[{CLIENT_ID}] ⚠️  NOTE: Models will NOT be saved during training")
    print(f"[{CLIENT_ID}] ⚠️  Final model will be available after all rounds complete\n")
    
    attempt = 0
    while attempt < MAX_RECONNECT_ATTEMPTS:
        try:
            if attempt > 0:
                print(f"\n[{CLIENT_ID}] 🔄 Retry {attempt + 1}/{MAX_RECONNECT_ATTEMPTS}")
                time.sleep(RECONNECT_DELAY)
            
            fl.client.start_client(
                server_address=server_address,
                client=client.to_client()
            )
            
            # Training completed!
            print(f"\n{'='*70}")
            print(f"[{CLIENT_ID}] 🎉 TRAINING COMPLETED!")
            print(f"{'='*70}")
            print(f"   Rounds Completed: {client.round_count}")
            print(f"   Final Accuracy: {client.final_accuracy:.4f}")
            print(f"   Final Loss: {client.final_loss:.4f}")
            
            # Save ONLY the final global model
            final_model_path = f"{CLIENT_ID}_FinalGlobal_v{len(update_notifier.notifications['pending_updates']) + 1}.h5"
            model.save(final_model_path)
            print(f"   💾 Final global model saved: {final_model_path}")
            
            # Create update notification
            notification = update_notifier.add_update_notification(
                total_rounds=client.round_count,
                final_accuracy=client.final_accuracy,
                final_loss=client.final_loss
            )
            
            print(f"\n{'='*70}")
            print(f"🔔 MODEL UPDATE AVAILABLE!")
            print(f"{'='*70}")
            print(f"   A new global model is ready for deployment")
            print(f"   Model File: {notification['model_file']}")
            print(f"   Final Accuracy: {notification['final_accuracy']:.4f}")
            print(f"   Final Loss: {notification['final_loss']:.4f}")
            print(f"   ")
            print(f"   ⚠️  ACTION REQUIRED:")
            print(f"   • Open the client menu")
            print(f"   • Select option 5 to view update details")
            print(f"   • Select option 6 to ACCEPT the update")
            print(f"   • Or select option 7 to REJECT the update")
            print(f"{'='*70}\n")
            
            # Show version history
            version_tracker.print_version_history()
            
            # Return to menu
            print(f"[{CLIENT_ID}] Returning to main menu...\n")
            time.sleep(2)
            
            # Restart menu to handle update
            main()
            
            break
            
        except KeyboardInterrupt:
            print(f"\n[{CLIENT_ID}] 🛑 Stopped")
            sys.exit(0)
            
        except Exception as e:
            attempt += 1
            print(f"[{CLIENT_ID}] ⚠️  Error: {str(e)[:80]}")
            if attempt >= MAX_RECONNECT_ATTEMPTS:
                print(f"\n[{CLIENT_ID}] ❌ Could not connect")
                sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[{CLIENT_ID}] ❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)