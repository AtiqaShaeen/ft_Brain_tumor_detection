"""
Asynchronous FL Server with Client Request System
- Clients send training requests
- Server shows pending requests
- Admin approves/rejects requests
- Multiple simultaneous trainings
- SAVES global model after each round for history/version tracking

Place in: server/server.py
"""

import flwr as fl
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import threading
import socket
import json
import time
import os
import sys
import numpy as np
import pickle

try:
    from fl_history_manager import FLHistoryManager
    from fl_version_control import ModelVersionControl
except ImportError:
    print("❌ Error: Required modules not found!")
    sys.exit(1)


# -------------------------------
# Training Request Queue
# -------------------------------
class TrainingRequestQueue:
    """Manages pending training requests from clients."""
    
    def __init__(self):
        self.pending_requests = {}  # {request_id: request_info}
        self.approved_requests = {}  # {client_id: approval_info}
        self.rejected_requests = []
        self.lock = threading.Lock()
        self.request_counter = 0
    
    def add_request(self, client_id: str, num_rounds: int, client_address: str):
        """Add a new training request from client."""
        with self.lock:
            self.request_counter += 1
            request_id = f"REQ_{self.request_counter}"
            
            request_info = {
                'request_id': request_id,
                'client_id': client_id,
                'num_rounds': num_rounds,
                'client_address': client_address,
                'timestamp': datetime.now(),
                'status': 'pending'
            }
            
            self.pending_requests[request_id] = request_info
            print(f"\n📨 NEW TRAINING REQUEST!")
            print(f"   Request ID: {request_id}")
            print(f"   Client ID: {client_id}")
            print(f"   Rounds: {num_rounds}")
            print(f"   From: {client_address}")
            print(f"   Time: {request_info['timestamp'].strftime('%H:%M:%S')}")
            print(f"   Status: PENDING APPROVAL\n")
            
            return request_id
    
    def approve_request(self, request_id: str, assigned_port: int):
        """Approve a training request."""
        with self.lock:
            if request_id in self.pending_requests:
                request = self.pending_requests.pop(request_id)
                request['status'] = 'approved'
                request['assigned_port'] = assigned_port
                request['approved_at'] = datetime.now()
                
                self.approved_requests[request['client_id']] = request
                return request
            return None
    
    def reject_request(self, request_id: str, reason: str = "Rejected by admin"):
        """Reject a training request."""
        with self.lock:
            if request_id in self.pending_requests:
                request = self.pending_requests.pop(request_id)
                request['status'] = 'rejected'
                request['reason'] = reason
                request['rejected_at'] = datetime.now()
                
                self.rejected_requests.append(request)
                return request
            return None
    
    def get_pending_requests(self) -> List[dict]:
        """Get list of pending requests."""
        with self.lock:
            return list(self.pending_requests.values())
    
    def print_pending_requests(self):
        """Display all pending requests."""
        with self.lock:
            if not self.pending_requests:
                print("\n✅ No pending training requests\n")
                return
            
            print(f"\n{'='*80}")
            print("📥 PENDING TRAINING REQUESTS")
            print(f"{'='*80}")
            print(f"{'Req ID':<12} {'Client ID':<15} {'Rounds':<8} {'Time':<12} {'From':<20}")
            print(f"{'-'*80}")
            
            for req in self.pending_requests.values():
                req_id = req['request_id']
                client_id = req['client_id']
                rounds = req['num_rounds']
                timestamp = req['timestamp'].strftime('%H:%M:%S')
                address = req['client_address']
                
                print(f"{req_id:<12} {client_id:<15} {rounds:<8} {timestamp:<12} {address:<20}")
            
            print(f"{'='*80}\n")


# -------------------------------
# Global Training State Manager
# -------------------------------
class TrainingStateManager:
    """Manages active training sessions."""
    
    def __init__(self):
        self.active_sessions = {}  # {client_id: session_info}
        self.lock = threading.Lock()
        self.training_threads = []
    
    def register_client(self, client_id: str, session_info: dict):
        """Register a new client training session."""
        with self.lock:
            self.active_sessions[client_id] = {
                **session_info,
                'status': 'training',
                'start_time': datetime.now(),
                'current_round': 0
            }
    
    def update_round(self, client_id: str, round_num: int):
        """Update current round for a client."""
        with self.lock:
            if client_id in self.active_sessions:
                self.active_sessions[client_id]['current_round'] = round_num
    
    def complete_training(self, client_id: str):
        """Mark client training as complete."""
        with self.lock:
            if client_id in self.active_sessions:
                self.active_sessions[client_id]['status'] = 'completed'
                self.active_sessions[client_id]['end_time'] = datetime.now()
    
    def get_active_clients(self) -> List[str]:
        """Get list of currently training clients."""
        with self.lock:
            return [cid for cid, info in self.active_sessions.items() 
                    if info['status'] == 'training']
    
    def get_session_info(self, client_id: str = None) -> dict:
        """Get session info."""
        with self.lock:
            if client_id:
                return self.active_sessions.get(client_id, {})
            return dict(self.active_sessions)
    
    def print_active_sessions(self):
        """Print all active training sessions."""
        with self.lock:
            if not self.active_sessions:
                print("\n✅ No active training sessions\n")
                return
            
            print(f"\n{'='*90}")
            print("📊 ACTIVE TRAINING SESSIONS")
            print(f"{'='*90}")
            print(f"{'Client ID':<15} {'Status':<12} {'Round':<10} {'Start':<12} {'Duration':<15}")
            print(f"{'-'*90}")
            
            for client_id, info in self.active_sessions.items():
                status = info['status']
                current_round = f"{info.get('current_round', 0)}/{info.get('total_rounds', '?')}"
                start_time = info['start_time'].strftime('%H:%M:%S')
                
                if status == 'training':
                    duration = (datetime.now() - info['start_time']).seconds
                    duration_str = f"{duration//60}m {duration%60}s"
                else:
                    end_time = info.get('end_time', info['start_time'])
                    duration = (end_time - info['start_time']).seconds
                    duration_str = f"{duration//60}m {duration%60}s"
                
                print(f"{client_id:<15} {status:<12} {current_round:<10} {start_time:<12} {duration_str:<15}")
            
            print(f"{'='*90}\n")


# -------------------------------
# Request Listener (Receives client requests)
# -------------------------------
class RequestListener:
    """Listens for incoming training requests from clients."""
    
    def __init__(self, request_queue: TrainingRequestQueue, port: int = 9090):
        self.request_queue = request_queue
        self.port = port
        self.running = True
        self.listener_thread = None
    
    def start(self):
        """Start listening for requests."""
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)
        self.listener_thread.start()
        print(f"✅ Request listener started on port {self.port}")
    
    def stop(self):
        """Stop the listener."""
        self.running = False
    
    def _listen(self):
        """Listen for incoming client requests."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(5)
        server_socket.settimeout(1.0)  # Check every second if still running
        
        print(f"👂 Listening for client requests on port {self.port}...")
        
        while self.running:
            try:
                client_socket, address = server_socket.accept()
                
                # Receive request data
                data = client_socket.recv(4096).decode('utf-8')
                request_data = json.loads(data)
                
                # Add to queue
                request_id = self.request_queue.add_request(
                    client_id=request_data['client_id'],
                    num_rounds=request_data['num_rounds'],
                    client_address=f"{address[0]}:{address[1]}"
                )
                
                # Send acknowledgment
                response = {
                    'status': 'received',
                    'request_id': request_id,
                    'message': 'Request received. Waiting for admin approval.'
                }
                client_socket.send(json.dumps(response).encode('utf-8'))
                client_socket.close()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"⚠️  Error receiving request: {e}")
        
        server_socket.close()


# -------------------------------
# Approval Notifier (Tells clients they're approved)
# -------------------------------
class ApprovalNotifier:
    """Notifies clients when their requests are approved."""
    
    def __init__(self, request_queue: TrainingRequestQueue):
        self.request_queue = request_queue
        self.running = True
        self.notifier_thread = None
    
    def start(self):
        """Start the notifier."""
        self.notifier_thread = threading.Thread(target=self._notify_loop, daemon=True)
        self.notifier_thread.start()
    
    def stop(self):
        """Stop the notifier."""
        self.running = False
    
    def _notify_loop(self):
        """Continuously check for approved requests and notify clients."""
        while self.running:
            time.sleep(1)
            # In a full implementation, this would send notifications
            # For now, clients will poll the server


# -------------------------------
# FL Strategy (FedProx with Client Drift Control)
# -------------------------------
class AsyncFedProx(FedProx):
    """Strategy for asynchronous federated learning with FedProx and client drift control."""
    
    def __init__(
        self,
        client_id: str,
        history_manager: FLHistoryManager,
        version_control: ModelVersionControl,
        training_state: TrainingStateManager,
        session_id: str,
        num_rounds: int,
        proximal_mu: float = 0.1,  # Proximal term coefficient for FedProx
        **kwargs
    ):
        super().__init__(proximal_mu=proximal_mu, **kwargs)
        self.client_id = client_id
        self.history_manager = history_manager
        self.version_control = version_control
        self.training_state = training_state
        self.session_id = session_id
        self.num_rounds = num_rounds
        self.current_round = 0
        self.last_parameters: Optional[Parameters] = None
        self.proximal_mu = proximal_mu

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self.current_round = server_round
        self.last_parameters = parameters
        self.training_state.update_round(self.client_id, server_round)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] [{self.client_id}] 🔄 Round {server_round}/{self.num_rounds}")
        print(f"[{timestamp}] [{self.client_id}] 🎯 FedProx - Proximal term µ={self.proximal_mu}")
        
        num_available = client_manager.num_available()
        if num_available < self.min_fit_clients:
            return []
        
        # Send proximal_mu to client for drift control
        config = {"proximal_mu": self.proximal_mu}
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        if not results:
            return self.last_parameters, {}
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{self.client_id}] 📦 Aggregating round {server_round} with FedProx")
        
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        self.last_parameters = aggregated_parameters
        
        # Store metrics but DON'T save model yet - only track for history
        if aggregated_parameters and metrics:
            accuracy = metrics.get("accuracy", 0.0)
            loss = metrics.get("loss", 0.0)
            
            print(f"[{timestamp}] [{self.client_id}] 🎯 Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            
            # Calculate client drift (if we have previous parameters)
            if self.last_parameters and server_round > 1:
                drift = self._calculate_client_drift(results)
                print(f"[{timestamp}] [{self.client_id}] 📊 Client Drift: {drift:.6f}")
                
                # Add drift to round data
                round_data = {
                    "client_id": self.client_id,
                    "accuracy": accuracy,
                    "loss": loss,
                    "num_clients": len(results),
                    "client_drift": drift,
                    "proximal_mu": self.proximal_mu
                }
            else:
                round_data = {
                    "client_id": self.client_id,
                    "accuracy": accuracy,
                    "loss": loss,
                    "num_clients": len(results),
                    "proximal_mu": self.proximal_mu
                }
            
            self.history_manager.add_round_data(server_round, round_data)
        
        return aggregated_parameters, metrics
    
    def _calculate_client_drift(self, results: List[Tuple[ClientProxy, FitRes]]) -> float:
        """
        Calculate client drift - measure how much clients deviated from global model.
        Returns the average L2 norm of parameter differences.
        """
        if not results or not self.last_parameters:
            return 0.0
        
        try:
            global_weights = [np.array(arr) for arr in self.last_parameters.tensors]
            
            total_drift = 0.0
            for _, fit_res in results:
                client_weights = [np.array(arr) for arr in fit_res.parameters.tensors]
                
                # Calculate L2 norm of difference
                drift = 0.0
                for g_w, c_w in zip(global_weights, client_weights):
                    drift += np.linalg.norm(g_w - c_w) ** 2
                
                total_drift += np.sqrt(drift)
            
            avg_drift = total_drift / len(results)
            return avg_drift
            
        except Exception as e:
            print(f"⚠️  Could not calculate drift: {e}")
            return 0.0

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ):
        if not results:
            return None, {}
        
        return super().aggregate_evaluate(server_round, results, failures)


# -------------------------------
# Start Training for Approved Client
# -------------------------------
def handle_client_training(
    client_id: str,
    num_rounds: int,
    assigned_port: int,
    history_manager: FLHistoryManager,
    version_control: ModelVersionControl,
    training_state: TrainingStateManager,
    proximal_mu: float = 0.1  # FedProx proximal term
):
    """Handle training for an approved client with FedProx."""
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting training for {client_id}")
    print(f"   Port: {assigned_port}")
    print(f"   Rounds: {num_rounds}")
    print(f"   Strategy: FedProx (µ={proximal_mu})")
    print(f"{'='*70}")
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "client_id": client_id,
        "num_rounds": num_rounds,
        "total_rounds": num_rounds,
        "session_id": session_id,
        "port": assigned_port,
        "strategy": "FedProx",
        "proximal_mu": proximal_mu
    }
    
    training_state.register_client(client_id, config)
    history_manager.start_new_session({**config, "address": f"0.0.0.0:{assigned_port}"})
    
    strategy = AsyncFedProx(
        client_id=client_id,
        history_manager=history_manager,
        version_control=version_control,
        training_state=training_state,
        session_id=session_id,
        num_rounds=num_rounds,
        proximal_mu=proximal_mu,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        accept_failures=True,
        initial_parameters=None,
    )
    
    try:
        fl.server.start_server(
            server_address=f"0.0.0.0:{assigned_port}",
            config=fl.server.ServerConfig(
                num_rounds=num_rounds,
                round_timeout=7200.0,
            ),
            strategy=strategy,
        )
        
        print(f"\n✅ {client_id} training completed!")
        print(f"   Total rounds: {num_rounds}")
        
        # NOW SAVE THE FINAL GLOBAL MODEL AFTER ALL TRAINING COMPLETES
        if strategy.last_parameters:
            print(f"   💾 Saving final global model...")
            
            # Convert parameters to numpy arrays
            weights = [np.array(arr) for arr in strategy.last_parameters.tensors]
            
            # Get final metrics from last round in current session
            final_metrics = {}
            avg_drift = 0.0
            if history_manager.current_session and history_manager.current_session.get('rounds'):
                last_round_data = history_manager.current_session['rounds'][-1]
                final_metrics = {
                    "accuracy": last_round_data.get('accuracy', 0.0),
                    "loss": last_round_data.get('loss', 0.0),
                    "proximal_mu": proximal_mu
                }
                
                # Calculate average drift across all rounds
                drifts = [r.get('client_drift', 0.0) for r in history_manager.current_session['rounds'] if 'client_drift' in r]
                if drifts:
                    avg_drift = sum(drifts) / len(drifts)
                    final_metrics["avg_client_drift"] = avg_drift
            
            # Save final model version
            version_num = version_control.save_model_version(
                model=weights,
                round_num=num_rounds,
                session_id=f"{session_id}_{client_id}",
                metrics=final_metrics,
                description=f"FINAL MODEL - FedProx (µ={proximal_mu}) - Client {client_id} - {num_rounds} rounds"
            )
            
            print(f"   ✅ Final global model saved as version {version_num}")
            print(f"   📊 Final Accuracy: {final_metrics.get('accuracy', 0.0):.4f}")
            print(f"   📊 Final Loss: {final_metrics.get('loss', 0.0):.4f}")
            if avg_drift > 0:
                print(f"   📊 Avg Client Drift: {avg_drift:.6f}")
        else:
            print(f"   ⚠️  Warning: No model parameters available to save")
        
        training_state.complete_training(client_id)
        history_manager.end_session(status="completed")
        
    except Exception as e:
        print(f"\n❌ {client_id} training failed: {e}")
        training_state.complete_training(client_id)
        history_manager.end_session(status="failed")


# -------------------------------
# Admin Menu
# -------------------------------
def show_admin_menu(
    request_queue: TrainingRequestQueue,
    training_state: TrainingStateManager,
    history_manager: FLHistoryManager,
    version_control: ModelVersionControl
) -> str:
    """Admin menu."""
    
    pending_count = len(request_queue.get_pending_requests())
    active_count = len(training_state.get_active_clients())
    
    print(f"\n{'='*70}")
    print("🎛️  FEDERATED LEARNING SERVER - ADMIN MENU")
    print(f"{'='*70}")
    print(f"📊 Status:")
    print(f"   📨 Pending Requests: {pending_count}")
    print(f"   🟢 Active Trainings: {active_count}")
    print(f"{'='*70}")
    print("1. 📥 View Pending Requests")
    print("2. ✅ Approve Training Request")
    print("3. ❌ Reject Training Request")
    print("4. 📊 View Active Training Sessions")
    print("5. 📜 View Training History")
    print("6. 📦 View Model Versions")
    print("7. 🔍 View Version Details")
    print("8. 📤 Export Version to Client")
    print("9. 📊 View Statistics")
    print("10. 🧹 Clean Up Old Versions")
    print("11. 🚪 Exit")
    print(f"{'='*70}")
    
    choice = input("\nEnter your choice (1-11): ").strip()
    return choice


# -------------------------------
# Main Server
# -------------------------------
def main():
    """Main server application."""
    
    history_manager = FLHistoryManager(history_dir="fl_history")
    version_control = ModelVersionControl(base_dir=".")
    training_state = TrainingStateManager()
    request_queue = TrainingRequestQueue()
    
    print("\n" + "="*70)
    print("🌟 ASYNCHRONOUS FL SERVER WITH REQUEST SYSTEM")
    print("="*70)
    print(f"📂 History: {os.path.abspath(history_manager.history_dir)}")
    print(f"📦 Models: {os.path.abspath(version_control.models_dir)}")
    print(f"📌 Current Version: v{version_control.current_version}")
    print("="*70)
    print("✅ Server listening for client requests")
    print("✅ Clients can request training anytime")
    print("✅ Admin approves/rejects requests")
    print("✅ Multiple simultaneous trainings supported")
    print("✅ Global model saved AFTER training completes")
    print("="*70)
    
    # Start request listener
    listener = RequestListener(request_queue, port=9090)
    listener.start()
    
    # Start approval notifier
    notifier = ApprovalNotifier(request_queue)
    notifier.start()
    
    # Port counter for assigning ports
    next_port = 8080
    
    # Main admin menu loop
    while True:
        try:
            choice = show_admin_menu(
                request_queue,
                training_state,
                history_manager,
                version_control
            )
            
            if choice == "1":
                request_queue.print_pending_requests()
                
            elif choice == "2":
                # Approve request
                request_queue.print_pending_requests()
                pending = request_queue.get_pending_requests()
                
                if not pending:
                    print("\n✅ No pending requests to approve\n")
                    continue
                
                req_id = input("\nEnter Request ID to approve: ").strip()
                
                # Ask for proximal_mu parameter
                # mu_input = input("Enter proximal_mu for FedProx (default: 0.1, higher = less drift): ").strip()
                # proximal_mu = float(mu_input) if mu_input else 0.1
                proximal_mu = 0.1
                
                # Approve and assign port
                request = request_queue.approve_request(req_id, next_port)
                
                if request:
                    print(f"\n✅ Request {req_id} APPROVED!")
                    print(f"   Client: {request['client_id']}")
                    print(f"   Assigned Port: {next_port}")
                    print(f"   Rounds: {request['num_rounds']}")
                    print(f"   Strategy: FedProx (µ={proximal_mu})")
                    
                    # Start training in background
                    training_thread = threading.Thread(
                        target=handle_client_training,
                        args=(
                            request['client_id'],
                            request['num_rounds'],
                            next_port,
                            history_manager,
                            version_control,
                            training_state,
                            proximal_mu  # Pass proximal_mu
                        ),
                        daemon=True
                    )
                    training_thread.start()
                    
                    print(f"   🚀 Training started in background")
                    print(f"   📡 Client should now connect to port {next_port}")
                    print(f"   💾 Final global model will be saved after all rounds complete\n")
                    
                    next_port += 1
                else:
                    print(f"\n❌ Request {req_id} not found\n")
            
            elif choice == "3":
                # Reject request
                request_queue.print_pending_requests()
                pending = request_queue.get_pending_requests()
                
                if not pending:
                    print("\n✅ No pending requests to reject\n")
                    continue
                
                req_id = input("\nEnter Request ID to reject: ").strip()
                reason = input("Reason (optional): ").strip() or "Rejected by admin"
                
                request = request_queue.reject_request(req_id, reason)
                
                if request:
                    print(f"\n❌ Request {req_id} REJECTED!")
                    print(f"   Client: {request['client_id']}")
                    print(f"   Reason: {reason}\n")
                else:
                    print(f"\n❌ Request {req_id} not found\n")
                    
            elif choice == "4":
                training_state.print_active_sessions()
                
            elif choice == "5":
                history_manager.view_history()
                
            elif choice == "6":
                version_control.print_versions()
                
            elif choice == "7":
                version = input("Enter version (Enter for latest): ").strip()
                version_num = int(version) if version.isdigit() else None
                info = version_control.get_version_info(version_num)
                if info:
                    print(f"\n{'='*70}")
                    print(f"📋 VERSION {info['version']} DETAILS")
                    print(f"{'='*70}")
                    for key, value in info.items():
                        print(f"   {key}: {value}")
                    print(f"{'='*70}\n")
                else:
                    print(f"\n❌ Version not found\n")
                    
            elif choice == "8":
                version_control.print_versions(limit=10)
                version = input("\nVersion to export (Enter for latest): ").strip()
                version_num = int(version) if version.isdigit() else None
                client_dir = input("Client directory (e.g., ../client1): ").strip()
                if client_dir:
                    version_control.export_version_to_client(version_num, client_dir)
                    
            elif choice == "9":
                hist_summary = history_manager.get_session_summary()
                ver_stats = version_control.get_stats()
                
                print(f"\n{'='*70}")
                print("📊 SYSTEM STATISTICS")
                print(f"{'='*70}")
                print("\n📜 Training History:")
                print(f"   Total Sessions: {hist_summary['total_sessions']}")
                print(f"   ✅ Completed: {hist_summary['completed']}")
                print(f"   ❌ Failed: {hist_summary['failed']}")
                print(f"   ⏸️  Interrupted: {hist_summary['interrupted']}")
                print("\n📦 Model Versions:")
                print(f"   Total: {ver_stats['total_versions']}")
                print(f"   Active: {ver_stats['active_versions']}")
                print(f"   Current: v{ver_stats['current_version']}")
                print(f"   Storage: {ver_stats['total_size_mb']:.2f} MB")
                print(f"   Location: {ver_stats['storage_path']}")
                print(f"{'='*70}\n")
                
            elif choice == "10":
                keep_n = input("\nKeep last N versions (default: 10): ").strip()
                keep_n = int(keep_n) if keep_n.isdigit() else 10
                version_control.cleanup_old_versions(keep_last_n=keep_n)
                
            elif choice == "11":
                active = training_state.get_active_clients()
                if active:
                    confirm = input(f"\n⚠️  {len(active)} client(s) still training! Exit? (yes/no): ")
                    if confirm.lower() != 'yes':
                        continue
                
                print("\n👋 Shutting down server...")
                listener.stop()
                notifier.stop()
                break
                
            else:
                print("\n❌ Invalid choice\n")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Ctrl+C detected")
            active = training_state.get_active_clients()
            if active:
                print(f"⚠️  {len(active)} client(s) still training!")
            confirm = input("Exit anyway? (yes/no): ")
            if confirm.lower() == 'yes':
                listener.stop()
                notifier.stop()
                break


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)