"""
Federated Learning Version Control System

Handles:
- Model version management (.h5 files)
- Version metadata tracking
- Client-server model distribution
- Version rollback capabilities

File: fl_version_control.py
"""

import os
import json
import shutil
from datetime import datetime
from typing import Optional, Dict, List, Any
import tensorflow as tf


class ModelVersionControl:
    """
    Manages model versions with .h5 file storage and metadata tracking.
    """
    
    def __init__(self, base_dir: str = "server"):
        """
        Initialize version control system.
        
        Args:
            base_dir: Base directory for server files (default: 'server')
        """
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "model_versions")
        self.metadata_file = os.path.join(base_dir, "version_metadata.json")
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
        self.current_version = self.metadata.get("current_version", 0)
    
    def _load_metadata(self) -> Dict:
        """Load version metadata from JSON file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._initialize_metadata()
        return self._initialize_metadata()
    
    def _initialize_metadata(self) -> Dict:
        """Initialize empty metadata structure."""
        return {
            "current_version": 0,
            "versions": {},
            "sessions": []
        }
    
    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError as e:
            print(f"❌ Error saving metadata: {e}")
    
    def save_model_version(
        self,
        model,
        round_num: int,
        session_id: str,
        metrics: Optional[Dict] = None,
        description: str = ""
    ) -> int:
        """
        Save a new model version as .h5 file.
        
        Args:
            model: TensorFlow/Keras model or model weights
            round_num: Training round number
            session_id: Session identifier
            metrics: Optional metrics dictionary
            description: Optional version description
            
        Returns:
            version_number: The version number assigned to this model
        """
        # Increment version
        self.current_version += 1
        version_num = self.current_version
        
        # Create filename
        model_filename = f"global_model_v{version_num}_session_{session_id}_round_{round_num}.h5"
        model_path = os.path.join(self.models_dir, model_filename)
        
        try:
            # Save model as .h5
            if hasattr(model, 'save'):
                # It's a Keras model
                model.save(model_path)
            else:
                # It's model weights - need to create a model structure
                print("⚠️  Warning: Saving weights only, model architecture not included")
                # For weights, we'll use pickle as fallback
                import pickle
                with open(model_path.replace('.h5', '.pkl'), 'wb') as f:
                    pickle.dump(model, f)
                model_filename = model_filename.replace('.h5', '.pkl')
                model_path = model_path.replace('.h5', '.pkl')
            
            # Create metadata entry
            version_metadata = {
                "version": version_num,
                "filename": model_filename,
                "filepath": model_path,
                "session_id": session_id,
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics or {},
                "description": description,
                "size_mb": os.path.getsize(model_path) / (1024 * 1024),
                "status": "active"
            }
            
            # Update metadata
            self.metadata["current_version"] = version_num
            self.metadata["versions"][str(version_num)] = version_metadata
            self._save_metadata()
            
            print(f"✅ Saved model version {version_num}: {model_filename}")
            return version_num
            
        except Exception as e:
            print(f"❌ Error saving model version: {e}")
            return -1
    
    def get_model_path(self, version: Optional[int] = None) -> Optional[str]:
        """
        Get the file path for a specific model version.
        
        Args:
            version: Version number (None = latest)
            
        Returns:
            Full path to the model file, or None if not found
        """
        if version is None:
            version = self.current_version
        
        version_key = str(version)
        if version_key not in self.metadata["versions"]:
            print(f"❌ Version {version} not found")
            return None
        
        model_path = self.metadata["versions"][version_key]["filepath"]
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None
        
        return model_path
    
    def load_model_version(self, version: Optional[int] = None):
        """
        Load a specific model version.
        
        Args:
            version: Version number (None = latest)
            
        Returns:
            Loaded model or None if error
        """
        model_path = self.get_model_path(version)
        if not model_path:
            return None
        
        try:
            if model_path.endswith('.h5'):
                model = tf.keras.models.load_model(model_path)
            else:
                # Load pickled weights
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            print(f"✅ Loaded model version {version or self.current_version}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def list_versions(self, limit: Optional[int] = None) -> List[Dict]:
        """
        List all available model versions.
        
        Args:
            limit: Maximum number of versions to return (None = all)
            
        Returns:
            List of version metadata dictionaries
        """
        versions = list(self.metadata["versions"].values())
        versions.sort(key=lambda x: x["version"], reverse=True)
        
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def print_versions(self, limit: int = 10) -> None:
        """Print formatted list of model versions."""
        versions = self.list_versions(limit)
        
        if not versions:
            print("\n❌ No model versions found.\n")
            return
        
        print(f"\n{'='*100}")
        print("📦 MODEL VERSIONS")
        print(f"{'='*100}")
        print(f"Current Version: v{self.current_version}\n")
        
        print(f"{'Ver':<6} {'Session':<18} {'Round':<8} {'Accuracy':<12} {'Size':<10} {'Timestamp':<20} {'Status':<10}")
        print(f"{'-'*100}")
        
        for v in versions:
            version = f"v{v['version']}"
            session = v['session_id'][:16] + "..." if len(v['session_id']) > 16 else v['session_id']
            round_num = v['round']
            accuracy = v['metrics'].get('accuracy', 'N/A')
            acc_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else accuracy
            size = f"{v['size_mb']:.2f} MB"
            timestamp = datetime.fromisoformat(v['timestamp']).strftime('%Y-%m-%d %H:%M')
            status = v['status']
            
            print(f"{version:<6} {session:<18} {round_num:<8} {acc_str:<12} {size:<10} {timestamp:<20} {status:<10}")
        
        print(f"{'='*100}")
        print(f"Total versions: {len(self.metadata['versions'])}")
        print(f"Storage: {os.path.abspath(self.models_dir)}")
        print(f"{'='*100}\n")
    
    def get_version_info(self, version: Optional[int] = None) -> Optional[Dict]:
        """
        Get detailed information about a specific version.
        
        Args:
            version: Version number (None = latest)
            
        Returns:
            Version metadata dictionary or None
        """
        if version is None:
            version = self.current_version
        
        version_key = str(version)
        return self.metadata["versions"].get(version_key)
    
    def delete_version(self, version: int, confirm: bool = False) -> bool:
        """
        Delete a specific model version.
        
        Args:
            version: Version number to delete
            confirm: Must be True to actually delete
            
        Returns:
            True if deleted successfully
        """
        if not confirm:
            print("⚠️  Set confirm=True to actually delete the version")
            return False
        
        version_key = str(version)
        if version_key not in self.metadata["versions"]:
            print(f"❌ Version {version} not found")
            return False
        
        # Don't allow deleting current version
        if version == self.current_version:
            print(f"❌ Cannot delete current version (v{version})")
            return False
        
        try:
            # Delete file
            model_path = self.metadata["versions"][version_key]["filepath"]
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # Update metadata
            self.metadata["versions"][version_key]["status"] = "deleted"
            self._save_metadata()
            
            print(f"✅ Deleted version {version}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting version: {e}")
            return False
    
    def rollback_to_version(self, version: int) -> bool:
        """
        Set a specific version as the current version.
        
        Args:
            version: Version number to rollback to
            
        Returns:
            True if successful
        """
        version_key = str(version)
        if version_key not in self.metadata["versions"]:
            print(f"❌ Version {version} not found")
            return False
        
        if self.metadata["versions"][version_key]["status"] != "active":
            print(f"❌ Version {version} is not active")
            return False
        
        self.current_version = version
        self.metadata["current_version"] = version
        self._save_metadata()
        
        print(f"✅ Rolled back to version {version}")
        return True
    
    def cleanup_old_versions(self, keep_last_n: int = 10) -> int:
        """
        Delete old model versions, keeping only the most recent ones.
        
        Args:
            keep_last_n: Number of recent versions to keep
            
        Returns:
            Number of versions deleted
        """
        versions = self.list_versions()
        
        if len(versions) <= keep_last_n:
            print(f"ℹ️  Only {len(versions)} versions found, nothing to delete")
            return 0
        
        # Delete older versions (excluding current)
        to_delete = [v for v in versions[keep_last_n:] if v["version"] != self.current_version]
        deleted_count = 0
        
        for v in to_delete:
            try:
                if os.path.exists(v["filepath"]):
                    os.remove(v["filepath"])
                self.metadata["versions"][str(v["version"])]["status"] = "deleted"
                deleted_count += 1
            except Exception as e:
                print(f"⚠️  Could not delete version {v['version']}: {e}")
        
        self._save_metadata()
        print(f"✅ Deleted {deleted_count} old version(s), kept {keep_last_n} most recent")
        return deleted_count
    
    def export_version_to_client(
        self,
        version: Optional[int] = None,
        client_dir: str = ".",
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Export a model version for client use.
        
        Args:
            version: Version number (None = latest)
            client_dir: Directory to export to
            filename: Custom filename (None = use default)
            
        Returns:
            Path to exported file or None if error
        """
        model_path = self.get_model_path(version)
        if not model_path:
            return None
        
        version_num = version or self.current_version
        
        # Determine export filename
        if filename is None:
            filename = f"global_model_v{version_num}.h5"
        
        export_path = os.path.join(client_dir, filename)
        
        try:
            shutil.copy2(model_path, export_path)
            print(f"✅ Exported version {version_num} to {export_path}")
            return export_path
        except Exception as e:
            print(f"❌ Error exporting model: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored versions."""
        versions = list(self.metadata["versions"].values())
        active_versions = [v for v in versions if v["status"] == "active"]
        
        total_size = sum(v["size_mb"] for v in active_versions)
        
        return {
            "total_versions": len(versions),
            "active_versions": len(active_versions),
            "current_version": self.current_version,
            "total_size_mb": total_size,
            "storage_path": os.path.abspath(self.models_dir)
        }


# Example usage
if __name__ == "__main__":
    mvc = ModelVersionControl()
    
    print("🧪 Testing Model Version Control\n")
    
    # Print stats
    stats = mvc.get_stats()
    print(f"📊 Stats: {stats}\n")
    
    # List versions
    mvc.print_versions()