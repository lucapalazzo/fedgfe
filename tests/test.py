import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from transformers import ViTModel, ViTConfig, ViTForImageClassification

import torch.nn as nn
import torch.optim as optim
import numpy as np

# Path to dataset splits
dataset_path = '/home/lpala/fedgfe/dataset/JSRT-8C-ClaSSeg'
dataset_path = '/home/lpala/fedgfe/dataset/ChestXRay-4N'

# Hyperparameters
batch_size = 32
learning_rate = 1e-5
num_epochs = 10
label_max_value = 0
label_count = 0

# torch seed bloccato
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # se si utilizza il multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset splits for each node
nodes_count = 4
dataset_splits = ['train', 'test']
# Assuming each node has its own directory with train/test splits
# Create a dictionary to hold DataLoader for each node
nodes = {}
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')

chosen_label = 0

class ClassificationHead(nn.Module):
    def __init__(self, backbone, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        x = x.last_hidden_state[:, 0, :]
        return self.fc(x)

for node_id in range(nodes_count):
    
    node_file = str(node_id) + ".npz"
    node_train_path = os.path.join(train_path, node_file)
    node_test_path = os.path.join(test_path, node_file)
    if not os.path.exists(node_train_path) or not os.path.exists(node_test_path):
        print(f"Node {node_id} does not have train/test data.")
        continue
    nodes[node_id] = {}

    nodes[node_id]['train_data'] = []
    nodes[node_id]['train_label'] = []
    nodes[node_id]['test_data'] = [] 
    nodes[node_id]['test_label'] = []
    nodes[node_id]['model'] = None
    # Load training data
    train_data = np.load(node_train_path, allow_pickle=True)['data'].tolist()
    test_data = np.load(node_test_path, allow_pickle=True)['data'].tolist()
    nodes[node_id]['train_data'] = torch.tensor(train_data['samples']).float()
    # get only first label per sample
    # nodes[node_id]['train_label'] = torch.tensor([label[chosen_label] for label in train_data['labels']]).long()
    nodes[node_id]['train_label'] = torch.tensor([label for label in train_data['labels']]).long()
    if nodes[node_id]['train_label'].max() > label_max_value:
        label_max_value = nodes[node_id]['train_label'].max()
    # Load testing data
    nodes[node_id]['test_data'] = torch.tensor(test_data['samples']).float()
    # nodes[node_id]['test_label'] = torch.tensor([label[chosen_label] for label in test_data['labels']]).long()
    nodes[node_id]['test_label'] = torch.tensor([label for label in test_data['labels']]).long()
    if nodes[node_id]['test_label'].max() >= label_max_value:
        label_max_value = nodes[node_id]['test_label'].max().item()
    # create DataLoader for training data

    label_count = label_max_value + 1
    train_dataset = torch.utils.data.TensorDataset(nodes[node_id]['train_data'], nodes[node_id]['train_label'])
    nodes[node_id]['train_loader'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # create DataLoader for testing data
    test_dataset = torch.utils.data.TensorDataset(nodes[node_id]['test_data'], nodes[node_id]['test_label'])
    nodes[node_id]['test_loader'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print (f"Node {node_id} - Train samples: {len(nodes[node_id]['train_data'])}, Test samples: {len(nodes[node_id]['test_data'])}")


    # check if sample and labels are the same for different nodes
for node_id in range(nodes_count):
    if torch.equal(nodes[node_id]['train_data'], nodes[node_id]['test_data']):
        print(f"Warning: Train data and test data for node {node_id} are the same")
    for target_node_id in range(nodes_count):
        if node_id == target_node_id:
            continue
        if torch.equal(nodes[node_id]['train_data'], nodes[target_node_id]['train_data']):
            print(f"Warning: Train data for node {node_id} is same of node {target_node_id}")
        if torch.equal(nodes[node_id]['train_label'], nodes[target_node_id]['train_label']):
            print(f"Warning: Train labels for node {node_id} is same of node {target_node_id}")
        if torch.equal(nodes[node_id]['test_data'], nodes[target_node_id]['test_data']):
            print(f"Warning: Test data for node {node_id} is same of node {target_node_id}")
        if torch.equal(nodes[node_id]['test_label'], nodes[target_node_id]['test_label']):
            print(f"Warning: Test labels for node {node_id} is same of node {target_node_id}")

# Define ResNet18 model
def create_model():
    vit_config = ViTConfig()
    vit_config.num_hidden_layers = 4
    # vit_config.loss_type = "cross_entropy"
    # vit_config.patch_size = 16
    # vit_config.output_attentions = True
    # vit_config.output_hidden_states = False
    # vit_config.num_hidden_layers = lf.args.num_hidden_layers
    vit_config.num_labels = label_count
    # backbone = ViTModel(vit_config)
    # model = ClassificationHead(backbone, vit_config.hidden_size, label_count)
    model = ViTForImageClassification(config=vit_config)
    # model = models.vit_b_16(pretrained=False)
    # model.heads[0] = nn.Linear(model.heads[0].in_features, label_count)  # Assuming 8 classes
    return model

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    step = 0
    epoch_loss = 0
    for images, labels in train_loader:
        step += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # logits = outputs
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state[:, 0, :]  # Use the first token's output for classification
        # logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / step

# Testing function
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # logits = outputs
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state[:, 0, :]  # Use the first token's output for classification
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# nodes = { 0: nodes[0] }
# Main loop for each node
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
round_count = 50
for round in range(round_count):
    federation_test_accuracy = []
    federation_train_accuracy = []
    # print(f"Round {round+1}/{round_count}")
# Iterate through each node and train/test the model
    for node, loaders in nodes.items():
        if nodes[node]['model'] is None:
            print(f"Creating model and optimizer for node {node}")
            model = create_model().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            # optimizer.add_param_group({'params': model.backbone.parameters(), 'lr': learning_rate})
            # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            nodes[node]['model'] = model
            nodes[node]['optimizer'] = optimizer

        model = nodes[node]['model']
        optimizer = nodes[node]['optimizer']
        # optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        for model_param, optimizer_param in zip( model.named_parameters(), optimizer.param_groups[0]['params'] ):
            if model_param[1] not in optimizer_param:
                print ( f"Warning: Model parameter {model_param[0]} not in optimizer parameter {optimizer_param}" )
            # if model_param[1].grad is not None:
            #     print ( f"Warning: Model parameter {model_param[0]} has gradient" )
            
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # print(f"Training and testing for node: {node} {hex(id(model))} {hex(id(optimizer))}")
        criterion = nn.CrossEntropyLoss()

        # Train the model
        round_loss = 0
        for epoch in range(num_epochs):
            round_loss += train_model(model, loaders['train_loader'], criterion, optimizer, device)
            # print(f"Epoch {epoch+1}/{num_epochs} completed for node {node}")

        # Test the model
        test_accuracy = test_model(model, loaders['test_loader'], device)
        train_accuracy = test_model(model, loaders['train_loader'], device)
        federation_test_accuracy.append(test_accuracy)
        federation_train_accuracy.append(train_accuracy)
        print(f"Round {round} Loss {round_loss/num_epochs:.4f} Accuracy for node {node}: test {test_accuracy * 100:.2f}% train {train_accuracy * 100:.2f}%")
    print(f"Federation accuracy for round {round}: test {np.mean(federation_test_accuracy) * 100:.2f}% train {np.mean(federation_train_accuracy) * 100:.2f}%")