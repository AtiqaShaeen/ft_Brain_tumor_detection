Federated Learning for Brain Tumor Classification 🧠
What is This Project?
This is a privacy-preserving brain tumor classification system where multiple hospitals or research centers can collaborate to build a better AI model without sharing patient data.
The Problem It Solves

Traditional AI: Hospital sends patient brain scans to a central server ❌
Our Solution: Hospitals keep data private, only share model improvements ✅

Real-World Example
Imagine three hospitals want to build a brain tumor detector:

Hospital A has 500 scans
Hospital B has 300 scans
Hospital C has 400 scans

Instead of combining all scans in one place (privacy risk!), each hospital:

Trains the AI on their own data locally
Sends only the "learning" (model weights) to a central server
Server combines the learnings
Sends improved AI back to all hospitals

Result: Everyone benefits from 1,200 total scans while data never leaves hospitals! 🎉

🏗️ System Architecture
┌─────────────────────────────────────────────────────────┐
│                   FEDERATED LEARNING                     │
│                                                          │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐│
│  │ Client 1 │◄────────┤  SERVER  ├────────►│ Client 2 ││
│  │          │  Model  │          │  Model  │          ││
│  │ Hospital │  Updates│   Admin  │  Updates│ Hospital ││
│  │    A     │         │  Control │         │    B     ││
│  └──────────┘         └──────────┘         └──────────┘│
│       ▲                                          ▲      │
│       │                                          │      │
│   Local Data                                 Local Data │
│   (Private)                                  (Private)  │
└─────────────────────────────────────────────────────────┘
Components
🖥️ Server (server.py)

Role: Central coordinator
What it does:

Receives training requests from clients
Allows admin to approve/reject requests
Aggregates model updates using FedProx algorithm
Tracks version history
Manages multiple simultaneous training sessions


Runs: On a central machine (can be cloud or local)

💻 Client 1 & Client 2 (client1.py, client2.py)

Role: Data owners (hospitals/institutions)
What they do:

Send training requests to server
Train AI model on local brain tumor data
Send only model improvements (not raw data!)
Receive updated global model
Get notifications when new models are ready


Runs: On hospital/institution computers


📋 Prerequisites
System Requirements

Python: 3.8 or higher
RAM: Minimum 8GB (16GB recommended)
GPU: Optional but recommended for faster training
Storage: ~5GB for dependencies + your dataset

Required Software
bash# Python packages (install via requirements.txt)
- tensorflow >= 2.10.0
- flwr >= 1.0.0
- numpy
- pillow
- scikit-learn

🚀 Installation Guide
Step 1: Install Python Dependencies
Create a virtual environment (recommended):
bash# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install required packages:
bashpip install -r requirements.txt
Step 2: Prepare Your Dataset
Your brain tumor dataset should be organized like this:
FEDERATED_BRAIN_TUMOR/
├── Braintumors_client1/
│   ├── glioma/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── Braintumors_client2/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── client1/
│   └── client1.py
│
├── client2/
│   └── client2.py
│
└── server/
    └── server.py
Step 3: Split Your Dataset
If you have one combined dataset, split it into two clients:
Option A: Manual Split (Simple)

Manually copy ~50% of images from each class to Braintumors_client1
Copy remaining ~50% to Braintumors_client2
Ensure both folders have all 4 tumor classes

Option B: Automated Split (Recommended)
Create split_dataset.py:
pythonimport os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, client1_dir, client2_dir, split_ratio=0.5):
    """
    Split dataset 50-50 between two clients.
    
    Args:
        source_dir: Original dataset folder (e.g., "Braintumors")
        client1_dir: Output folder for client 1
        client2_dir: Output folder for client 2
        split_ratio: How much data goes to client 2 (0.5 = 50-50 split)
    """
    print(f"📂 Starting dataset split...")
    print(f"   Source: {source_dir}")
    print(f"   Client 1: {client1_dir}")
    print(f"   Client 2: {client2_dir}")
    
    os.makedirs(client1_dir, exist_ok=True)
    os.makedirs(client2_dir, exist_ok=True)
    
    total_images = 0
    client1_count = 0
    client2_count = 0
    
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        print(f"\n📁 Processing class: {class_name}")
        
        # Get all images
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        total_images += len(images)
        print(f"   Found {len(images)} images")
        
        # Split images
        client1_images, client2_images = train_test_split(
            images, 
            test_size=split_ratio, 
            random_state=42
        )
        
        client1_count += len(client1_images)
        client2_count += len(client2_images)
        
        # Create class directories
        os.makedirs(os.path.join(client1_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(client2_dir, class_name), exist_ok=True)
        
        # Copy files to client1
        for img in client1_images:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(client1_dir, class_name, img)
            )
        
        # Copy files to client2
        for img in client2_images:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(client2_dir, class_name, img)
            )
        
        print(f"   ✅ Client 1: {len(client1_images)} images")
        print(f"   ✅ Client 2: {len(client2_images)} images")
    
    print(f"\n{'='*60}")
    print(f"✅ Dataset split completed!")
    print(f"{'='*60}")
    print(f"📊 Summary:")
    print(f"   Total images: {total_images}")
    print(f"   Client 1: {client1_count} images ({client1_count/total_images*100:.1f}%)")
    print(f"   Client 2: {client2_count} images ({client2_count/total_images*100:.1f}%)")
    print(f"{'='*60}")

# Run the split
if __name__ == "__main__":
    # Adjust these paths to match your setup
    split_dataset(
        source_dir="Braintumors",           # Your original dataset
        client1_dir="Braintumors_client1",  # Client 1 will get this
        client2_dir="Braintumors_client2"   # Client 2 will get this
    )
Run it:
bashpython split_dataset.py

🎮 How to Use (Step-by-Step)
Scenario: Two Hospitals Training Together
Step 1: Start the Server (Admin)
Open a terminal in the server folder:
bashcd server
python server.py
You'll see:
======================================================================
🌟 ASYNCHRONOUS FL SERVER WITH REQUEST SYSTEM
======================================================================
📂 History: /path/to/fl_history
📦 Models: /path/to/model_versions
📌 Current Version: v0
======================================================================
✅ Server listening for client requests
✅ Clients can request training anytime
✅ Admin approves/rejects requests
✅ Multiple simultaneous trainings supported
✅ Global model saved AFTER training completes
======================================================================
The server will display an admin menu:
======================================================================
🎛️  FEDERATED LEARNING SERVER - ADMIN MENU
======================================================================
📊 Status:
   📨 Pending Requests: 0
   🟢 Active Trainings: 0
======================================================================
1. 📥 View Pending Requests
2. ✅ Approve Training Request
3. ❌ Reject Training Request
4. 📊 View Active Training Sessions
5. 📜 View Training History
6. 📦 View Model Versions
7. 🔍 View Version Details
8. 📤 Export Version to Client
9. 📊 View Statistics
10. 🧹 Clean Up Old Versions
11. 🚪 Exit
======================================================================
Keep this terminal open!

Step 2: Start Client 1 (Hospital A)
Open a new terminal in the client1 folder:
bashcd client1
python client1.py
You'll see a client menu:
======================================================================
🚀 FEDERATED LEARNING CLIENT
======================================================================
📋 Client ID: client1
📂 Dataset: Braintumors_client1
======================================================================

======================================================================
[client1] 🎛️  CLIENT MENU
======================================================================
1. 🚀 Send Training Request to Server
2. 📜 View Version History
3. 🔍 View Last Version
4. 📂 List Local Models
5. 🔔 View Model Update Notifications
6. ✅ Accept Model Update
7. ❌ Reject Model Update
8. ⚙️  Settings
9. 🚪 Exit
======================================================================
Select option 1 to send a training request:
Enter your choice (1-9): 1
Fill in the details:
Server host (default: 127.0.0.1): [Press Enter]
Request port (default: 9090): [Press Enter]
Number of rounds (default: 10): 5
Expected FL port (default: 8080): [Press Enter]
Client 1 will send the request and wait for approval:
[client1] ✅ Request sent successfully!
[client1] 📋 Request ID: REQ_1
[client1] 💬 Request received. Waiting for admin approval.

⚠️  IMPORTANT:
   1. Admin must approve your request on the server
   2. Admin should select your Request ID: REQ_1
   3. Once approved, training will start automatically
   
⏳ Waiting for admin approval...
   Checking if port 8080 is ready...

Step 3: Start Client 2 (Hospital B)
Open another new terminal in the client2 folder:
bashcd client2
python client2.py
Repeat the same process:

Select option 1
Fill in the same server details
Request 5 rounds
Use port 8081 (different from Client 1!)

Client 2 will also wait for approval.

Step 4: Admin Approves Requests
Back in the server terminal, you'll see notifications:
📨 NEW TRAINING REQUEST!
   Request ID: REQ_1
   Client ID: client1
   Rounds: 5
   From: 127.0.0.1:xxxxx
   Time: 14:30:25
   Status: PENDING APPROVAL

📨 NEW TRAINING REQUEST!
   Request ID: REQ_2
   Client ID: client2
   Rounds: 5
   From: 127.0.0.1:xxxxx
   Time: 14:30:45
   Status: PENDING APPROVAL
Select option 1 to view pending requests:
Enter your choice (1-11): 1
You'll see:
================================================================================
📥 PENDING TRAINING REQUESTS
================================================================================
Req ID       Client ID       Rounds   Time         From                
--------------------------------------------------------------------------------
REQ_1        client1         5        14:30:25     127.0.0.1:xxxxx     
REQ_2        client2         5        14:30:45     127.0.0.1:xxxxx     
================================================================================
Select option 2 to approve Client 1:
Enter your choice (1-11): 2
Enter Request ID to approve: REQ_1
Server will start training for Client 1:
✅ Request REQ_1 APPROVED!
   Client: client1
   Assigned Port: 8080
   Rounds: 5
   Strategy: FedProx (µ=0.1)
   🚀 Training started in background
Approve Client 2 the same way:
Enter your choice (1-11): 2
Enter Request ID to approve: REQ_2

Step 5: Training Happens Automatically
Now watch the magic happen! In the client terminals, you'll see:
Client 1:
✅ Request APPROVED! Port 8080 is ready.

[client1] 📂 Loading dataset...
[client1] 📊 Dataset:
   Classes: ['glioma', 'meningioma', 'notumor', 'pituitary']
   Training: 2400
   Validation: 600

[client1] 🧠 Building model...
[client1] ✅ Model ready

[client1] 🌐 Connecting to 127.0.0.1:8080...
[client1] ⚠️  NOTE: Models will NOT be saved during training
[client1] ⚠️  Final model will be available after all rounds complete

======================================================================
[client1] 🔄 ROUND 1
======================================================================
[client1] 📥 Received global model
[client1] 🧠 Training 5 epochs on 2400 samples
Epoch 1/5
75/75 [==============================] - 45s 600ms/step - loss: 1.2345 - accuracy: 0.6789
...
[client1] ✅ Completed in 3.5 min
[client1] 📊 Acc: 0.7234, Loss: 0.8901
[client1] 📤 Sending updates...
======================================================================
Server will show aggregation:
[14:32:15] [client1] 🔄 Round 1/5
[14:32:15] [client1] 🎯 FedProx - Proximal term µ=0.1
[14:35:42] [client1] 📦 Aggregating round 1 with FedProx
[14:35:42] [client1] 🎯 Accuracy: 0.7234, Loss: 0.8901
This continues for all 5 rounds!

Step 6: Training Completes
After 5 rounds, clients will see:
======================================================================
[client1] 🎉 TRAINING COMPLETED!
======================================================================
   Rounds Completed: 5
   Final Accuracy: 0.8956
   Final Loss: 0.3421
   💾 Final global model saved: client1_FinalGlobal_v1.h5

======================================================================
🔔 MODEL UPDATE AVAILABLE!
======================================================================
   A new global model is ready for deployment
   Model File: client1_FinalGlobal_v1.h5
   Final Accuracy: 0.8956
   Final Loss: 0.3421
   
   ⚠️  ACTION REQUIRED:
   • Open the client menu
   • Select option 5 to view update details
   • Select option 6 to ACCEPT the update
   • Or select option 7 to REJECT the update
======================================================================

Step 7: Client Reviews and Accepts Model
Client returns to menu. Select option 5 to view the update:
Enter your choice (1-9): 5
Shows details:
======================================================================
🔔 MODEL UPDATE AVAILABLE!
======================================================================

Update #1:
   Timestamp: 2024-01-09 14:45:30
   Training Rounds: 5
   Final Accuracy: 0.8956
   Final Loss: 0.3421
   Model File: client1_FinalGlobal_v1.h5
   Status: PENDING
======================================================================
Select option 6 to accept:
Enter your choice (1-9): 6
Enter update number to accept (or 0 to cancel): 1

✅ Update accepted!
   Model file: client1_FinalGlobal_v1.h5
   You can now use this model for inference/deployment
Now the model is ready for production use! 🎉

📊 Understanding the Output
What Gets Saved?
On Server
server/
├── fl_history/              # Training session logs
│   └── session_*.json
├── model_versions/          # All model versions
│   ├── version_1/
│   │   ├── model.pkl
│   │   └── metadata.json
│   └── version_2/
└── version_metadata.json    # Version tracking
On Each Client
client1/
├── client1_FinalGlobal_v1.h5           # Final trained model
├── client_version_log.json              # Training history
└── model_update_notification.json       # Pending updates
Model Naming Convention

client1_FinalGlobal_v1.h5: Final global model after all rounds
The number (v1, v2, etc.) increments with each completed training
