"""
Federated Learning History and Version Management Module

This module provides comprehensive tracking and management of:
- Training session history
- Round-by-round metrics
- Model version control
- Persistent storage

Author: Your Name
Date: 2025-01-04
"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any


class FLHistoryManager:
    """
    Manages federated learning training history and model versions.
    
    Features:
    - Session tracking with timestamps
    - Round-by-round metrics storage
    - Model version control
    - JSON-based persistent storage
    """
    
    def __init__(self, history_dir: str = "fl_history"):
        """
        Initialize the history manager.
        
        Args:
            history_dir: Directory to store history and models
        """
        self.history_dir = history_dir
        self.history_file = os.path.join(history_dir, "training_history.json")
        self.models_dir = os.path.join(history_dir, "models")
        
        # Create directories if they don't exist
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load existing history
        self.history = self._load_history()
        self.current_session: Optional[Dict] = None
    
    def _load_history(self) -> Dict:
        """Load training history from JSON file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Warning: Could not load history file: {e}")
                return {"sessions": []}
        return {"sessions": []}
    
    def _save_history(self) -> None:
        """Save training history to JSON file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except IOError as e:
            print(f"❌ Error saving history: {e}")
    
    def start_new_session(self, config: Dict[str, Any]) -> str:
        """
        Start a new training session.
        
        Args:
            config: Configuration dictionary for the session
            
        Returns:
            session_id: Unique identifier for this session
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "config": config,
            "rounds": [],
            "status": "running"
        }
        return session_id
    
    def add_round_data(self, round_num: int, round_data: Dict[str, Any]) -> None:
        """
        Add data for a completed round.
        
        Args:
            round_num: Round number
            round_data: Dictionary containing round metrics
        """
        if self.current_session:
            self.current_session["rounds"].append({
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                **round_data
            })
            self._save_history()
    
    def end_session(self, status: str = "completed") -> None:
        """
        End the current training session.
        
        Args:
            status: Final status of the session (completed, interrupted, failed)
        """
        if self.current_session:
            self.current_session["end_time"] = datetime.now().isoformat()
            self.current_session["status"] = status
            self.history["sessions"].append(self.current_session)
            self._save_history()
            self.current_session = None
    
    def save_model_version(
        self, 
        parameters: Any, 
        round_num: int, 
        metrics: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Save model parameters as a versioned file.
        
        Args:
            parameters: Model parameters to save
            round_num: Round number
            metrics: Optional metrics dictionary
            
        Returns:
            Path to saved model file, or None if no current session
        """
        if not self.current_session:
            return None
        
        session_id = self.current_session["session_id"]
        model_filename = f"global_model_{session_id}_round_{round_num}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        try:
            # Save model parameters with metadata
            with open(model_path, 'wb') as f:
                pickle.dump({
                    "parameters": parameters,
                    "round": round_num,
                    "session_id": session_id,
                    "metrics": metrics or {},
                    "timestamp": datetime.now().isoformat()
                }, f)
            
            return model_path
        except IOError as e:
            print(f"❌ Error saving model: {e}")
            return None
    
    def load_model_version(self, model_filename: str) -> Optional[Dict]:
        """
        Load a specific model version.
        
        Args:
            model_filename: Name of the model file to load
            
        Returns:
            Dictionary containing model data, or None if file not found
        """
        model_path = os.path.join(self.models_dir, model_filename)
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_filename}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, IOError) as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def view_history(self) -> None:
        """Display formatted training history."""
        if not self.history["sessions"]:
            print("\n❌ No training history found.\n")
            return
        
        print(f"\n{'='*90}")
        print("📜 FEDERATED LEARNING TRAINING HISTORY")
        print(f"{'='*90}\n")
        
        for i, session in enumerate(reversed(self.history["sessions"]), 1):
            self._print_session_details(i, session)
    
    def _print_session_details(self, index: int, session: Dict) -> None:
        """Print details for a single session."""
        print(f"Session {index}: {session['session_id']}")
        print(f"{'─'*90}")
        print(f"   Status: {session['status'].upper()}")
        print(f"   Start Time: {self._format_datetime(session['start_time'])}")
        
        if session['end_time']:
            print(f"   End Time: {self._format_datetime(session['end_time'])}")
            duration = self._calculate_duration(session['start_time'], session['end_time'])
            print(f"   Duration: {duration}")
        
        print(f"   Total Rounds: {len(session['rounds'])}")
        print(f"   Configuration:")
        for key, value in session['config'].items():
            print(f"      • {key}: {value}")
        
        if session['rounds']:
            self._print_round_table(session['rounds'])
        
        print(f"\n{'='*90}\n")
    
    def _print_round_table(self, rounds: List[Dict]) -> None:
        """Print a formatted table of round details."""
        print(f"\n   Round Details:")
        print(f"   {'Round':<8} {'Timestamp':<20} {'Accuracy':<12} {'Loss':<12} {'Clients':<10}")
        print(f"   {'-'*70}")
        
        for round_data in rounds:
            round_num = round_data['round']
            timestamp = self._format_time(round_data['timestamp'])
            accuracy = round_data.get('accuracy', 'N/A')
            loss = round_data.get('loss', 'N/A')
            clients = round_data.get('num_clients', 'N/A')
            
            acc_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else accuracy
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else loss
            
            print(f"   {round_num:<8} {timestamp:<20} {acc_str:<12} {loss_str:<12} {clients:<10}")
    
    def view_model_versions(self) -> None:
        """Display all saved model versions."""
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            print("\n❌ No model versions found.\n")
            return
        
        print(f"\n{'='*90}")
        print("📦 SAVED MODEL VERSIONS")
        print(f"{'='*90}\n")
        
        # Sort by modification time (newest first)
        model_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.models_dir, x)), 
            reverse=True
        )
        
        print(f"{'#':<5} {'Filename':<45} {'Size':<12} {'Date':<20}")
        print(f"{'-'*90}")
        
        for i, filename in enumerate(model_files, 1):
            filepath = os.path.join(self.models_dir, filename)
            size = os.path.getsize(filepath)
            size_mb = size / (1024 * 1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            print(f"{i:<5} {filename:<45} {size_mb:>8.2f} MB   {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n{'='*90}")
        print(f"Total models: {len(model_files)}")
        print(f"Storage location: {os.path.abspath(self.models_dir)}")
        print(f"{'='*90}\n")
    
    def get_latest_model(self) -> Optional[str]:
        """
        Get the filename of the most recent model.
        
        Returns:
            Filename of the latest model, or None if no models exist
        """
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            return None
        
        # Sort by modification time (newest first)
        model_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.models_dir, x)), 
            reverse=True
        )
        
        return model_files[0]
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all sessions.
        
        Returns:
            Dictionary containing session statistics
        """
        total_sessions = len(self.history["sessions"])
        completed = sum(1 for s in self.history["sessions"] if s["status"] == "completed")
        failed = sum(1 for s in self.history["sessions"] if s["status"] == "failed")
        interrupted = sum(1 for s in self.history["sessions"] if s["status"] == "interrupted")
        
        return {
            "total_sessions": total_sessions,
            "completed": completed,
            "failed": failed,
            "interrupted": interrupted
        }
    
    def clear_history(self, confirm: bool = False) -> bool:
        """
        Clear all training history (WARNING: This is irreversible!).
        
        Args:
            confirm: Must be True to actually clear history
            
        Returns:
            True if history was cleared, False otherwise
        """
        if not confirm:
            print("⚠️  Warning: Set confirm=True to actually clear history")
            return False
        
        self.history = {"sessions": []}
        self._save_history()
        print("✅ Training history cleared")
        return True
    
    def delete_old_models(self, keep_last_n: int = 10) -> int:
        """
        Delete old model versions, keeping only the most recent ones.
        
        Args:
            keep_last_n: Number of recent models to keep
            
        Returns:
            Number of models deleted
        """
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        
        if len(model_files) <= keep_last_n:
            print(f"ℹ️  Only {len(model_files)} models found, nothing to delete")
            return 0
        
        # Sort by modification time (oldest first for deletion)
        model_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.models_dir, x))
        )
        
        # Delete oldest models
        to_delete = model_files[:-keep_last_n]
        deleted_count = 0
        
        for filename in to_delete:
            try:
                os.remove(os.path.join(self.models_dir, filename))
                deleted_count += 1
            except OSError as e:
                print(f"⚠️  Could not delete {filename}: {e}")
        
        print(f"✅ Deleted {deleted_count} old model(s), kept {keep_last_n} most recent")
        return deleted_count
    
    # Helper methods for formatting
    def _format_datetime(self, iso_string: str) -> str:
        """Format ISO datetime string to readable format."""
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def _format_time(self, iso_string: str) -> str:
        """Format ISO datetime to time only."""
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%H:%M:%S")
    
    def _calculate_duration(self, start_iso: str, end_iso: str) -> str:
        """Calculate duration between two ISO datetime strings."""
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        duration = end - start
        
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


# Example usage and testing
if __name__ == "__main__":
    # Create a history manager instance
    manager = FLHistoryManager()
    
    print("🧪 Testing FL History Manager\n")
    
    # Example: Start a session
    config = {
        "num_rounds": 5,
        "min_clients": 2,
        "learning_rate": 0.001
    }
    session_id = manager.start_new_session(config)
    print(f"✅ Started session: {session_id}")
    
    # Example: Add some round data
    for round_num in range(1, 4):
        round_data = {
            "accuracy": 0.8 + round_num * 0.05,
            "loss": 0.5 - round_num * 0.1,
            "num_clients": 2
        }
        manager.add_round_data(round_num, round_data)
        print(f"✅ Added data for round {round_num}")
    
    # Example: End the session
    manager.end_session(status="completed")
    print(f"✅ Ended session\n")
    
    # View history
    manager.view_history()
    
    # Get summary
    summary = manager.get_session_summary()
    print(f"📊 Summary: {summary}")