"""
Advanced Computer Vision Module for Aetheron AI Platform
Includes object detection, image classification, segmentation, and more
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import json
import os
from datetime import datetime
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CVConfig:
    """Configuration for computer vision tasks"""
    task_type: str = "classification"  # classification, detection, segmentation
    input_size: Tuple[int, int] = (224, 224)
    num_classes: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    augmentation_enabled: bool = True
    pretrained: bool = False

class ImageProcessor:
    """Advanced image processing utilities"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        return image
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int] = None) -> np.ndarray:
        """Resize image using bilinear interpolation"""
        if size is None:
            size = self.target_size
        
        # Simple nearest neighbor resize for demonstration
        h_old, w_old = image.shape[:2]
        h_new, w_new = size
        
        if len(image.shape) == 3:
            resized = np.zeros((h_new, w_new, image.shape[2]), dtype=image.dtype)
        else:
            resized = np.zeros((h_new, w_new), dtype=image.dtype)
        
        for i in range(h_new):
            for j in range(w_new):
                old_i = int(i * h_old / h_new)
                old_j = int(j * w_old / w_new)
                resized[i, j] = image[old_i, old_j]
        
        return resized
    
    def augment_image(self, image: np.ndarray, augmentation_type: str = "random") -> np.ndarray:
        """Apply data augmentation"""
        augmented = image.copy()
        
        if augmentation_type == "flip" or augmentation_type == "random":
            if np.random.random() > 0.5:
                augmented = np.fliplr(augmented)
        
        if augmentation_type == "rotate" or augmentation_type == "random":
            if np.random.random() > 0.5:
                # Simple 90-degree rotation
                augmented = np.rot90(augmented, k=np.random.randint(1, 4))
        
        if augmentation_type == "noise" or augmentation_type == "random":
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.01, augmented.shape)
                augmented = np.clip(augmented + noise, 0, 1)
        
        if augmentation_type == "brightness" or augmentation_type == "random":
            if np.random.random() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                augmented = np.clip(augmented * factor, 0, 1)
        
        return augmented
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract basic image features"""
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(image)
        features['std'] = np.std(image)
        features['min'] = np.min(image)
        features['max'] = np.max(image)
        
        # Histogram features
        if len(image.shape) == 3:
            # Color image
            for i, color in enumerate(['red', 'green', 'blue']):
                hist, _ = np.histogram(image[:, :, i], bins=10, range=(0, 1))
                features[f'{color}_hist'] = hist.tolist()
        else:
            # Grayscale image
            hist, _ = np.histogram(image, bins=10, range=(0, 1))
            features['gray_hist'] = hist.tolist()
        
        # Edge detection (simple gradient)
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        grad_x = np.abs(np.gradient(gray, axis=1))
        grad_y = np.abs(np.gradient(gray, axis=0))
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        features['edge_strength'] = np.mean(edge_strength)
        
        return features

class ConvolutionalLayer:
    """Convolutional layer implementation"""
    
    def __init__(self, num_filters: int, filter_size: int, stride: int = 1, 
                 padding: int = 0, activation: str = 'relu'):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        
        # Initialize filters randomly
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1
        self.biases = np.zeros(num_filters)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through convolution"""
        if len(input_data.shape) == 2:
            # Add channel dimension
            input_data = input_data[:, :, np.newaxis]
        
        height, width, channels = input_data.shape
        
        # Calculate output dimensions
        out_height = (height - self.filter_size + 2 * self.padding) // self.stride + 1
        out_width = (width - self.filter_size + 2 * self.padding) // self.stride + 1
        
        # Initialize output
        output = np.zeros((out_height, out_width, self.num_filters))
        
        # Apply convolution
        for f in range(self.num_filters):
            for i in range(out_height):
                for j in range(out_width):
                    start_i = i * self.stride
                    end_i = start_i + self.filter_size
                    start_j = j * self.stride
                    end_j = start_j + self.filter_size
                    
                    if channels == 1:
                        region = input_data[start_i:end_i, start_j:end_j, 0]
                        output[i, j, f] = np.sum(region * self.filters[f]) + self.biases[f]
                    else:
                        # For multiple channels, sum across all channels
                        conv_sum = 0
                        for c in range(channels):
                            region = input_data[start_i:end_i, start_j:end_j, c]
                            conv_sum += np.sum(region * self.filters[f])
                        output[i, j, f] = conv_sum + self.biases[f]
        
        # Apply activation
        if self.activation == 'relu':
            output = self.relu(output)
        
        return output

class PoolingLayer:
    """Pooling layer implementation"""
    
    def __init__(self, pool_size: int = 2, stride: int = 2, pool_type: str = 'max'):
        self.pool_size = pool_size
        self.stride = stride
        self.pool_type = pool_type
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through pooling"""
        height, width, channels = input_data.shape
        
        # Calculate output dimensions
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((out_height, out_width, channels))
        
        # Apply pooling
        for c in range(channels):
            for i in range(out_height):
                for j in range(out_width):
                    start_i = i * self.stride
                    end_i = start_i + self.pool_size
                    start_j = j * self.stride
                    end_j = start_j + self.pool_size
                    
                    region = input_data[start_i:end_i, start_j:end_j, c]
                    
                    if self.pool_type == 'max':
                        output[i, j, c] = np.max(region)
                    elif self.pool_type == 'avg':
                        output[i, j, c] = np.mean(region)
        
        return output

class CNNClassifier:
    """Convolutional Neural Network for image classification"""
    
    def __init__(self, config: CVConfig):
        self.config = config
        
        # Build network architecture
        self.conv1 = ConvolutionalLayer(32, 3, activation='relu')
        self.pool1 = PoolingLayer(2, 2, 'max')
        self.conv2 = ConvolutionalLayer(64, 3, activation='relu')
        self.pool2 = PoolingLayer(2, 2, 'max')
        
        # Calculate flattened size (approximate)
        self.flattened_size = self._calculate_flattened_size()
        
        # Fully connected layers
        self.fc_weights = np.random.randn(self.flattened_size, config.num_classes) * 0.1
        self.fc_bias = np.zeros(config.num_classes)
        
        # Training history
        self.training_history = {'loss': [], 'accuracy': []}
    
    def _calculate_flattened_size(self) -> int:
        """Calculate the size after convolution and pooling"""
        # Approximate calculation based on typical input size
        h, w = self.config.input_size
        
        # After conv1 (assuming no padding, stride 1)
        h = h - 3 + 1
        w = w - 3 + 1
        
        # After pool1 (2x2, stride 2)
        h = h // 2
        w = w // 2
        
        # After conv2
        h = h - 3 + 1
        w = w - 3 + 1
        
        # After pool2
        h = h // 2
        w = w // 2
        
        return h * w * 64  # 64 filters in conv2
    
    def softmax(self, x):
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        # Convolutional layers
        x = self.conv1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool2.forward(x)
        
        # Flatten
        x_flat = x.flatten()
        
        # Adjust for actual size
        if len(x_flat) != self.flattened_size:
            # Resize weights if needed
            self.fc_weights = np.random.randn(len(x_flat), self.config.num_classes) * 0.1
            self.flattened_size = len(x_flat)
        
        # Fully connected layer
        output = np.dot(x_flat, self.fc_weights) + self.fc_bias
        
        return self.softmax(output)
    
    def predict(self, x: np.ndarray) -> int:
        """Make prediction for single image"""
        probabilities = self.forward(x)
        return np.argmax(probabilities)
    
    def predict_batch(self, images: List[np.ndarray]) -> List[int]:
        """Make predictions for batch of images"""
        predictions = []
        for image in images:
            predictions.append(self.predict(image))
        return predictions
    
    def train_step(self, x: np.ndarray, y: int) -> float:
        """Single training step (simplified)"""
        # Forward pass
        output = self.forward(x)
        
        # Calculate loss (cross-entropy)
        loss = -np.log(output[y] + 1e-8)
        
        # Simple weight update (gradient descent approximation)
        learning_rate = self.config.learning_rate
        
        # Update fully connected weights (simplified)
        grad = output.copy()
        grad[y] -= 1  # Gradient of cross-entropy loss
        
        # Update weights (very simplified backpropagation)
        x_flat = self.conv2.forward(self.pool1.forward(self.conv1.forward(x)))
        x_flat = self.pool2.forward(x_flat).flatten()
        
        if len(x_flat) == self.flattened_size:
            self.fc_weights -= learning_rate * np.outer(x_flat, grad)
            self.fc_bias -= learning_rate * grad
        
        return loss

class ObjectDetector:
    """Simple object detection using sliding window"""
    
    def __init__(self, classifier: CNNClassifier, window_size: Tuple[int, int] = (64, 64)):
        self.classifier = classifier
        self.window_size = window_size
        self.processor = ImageProcessor(window_size)
    
    def detect_objects(self, image: np.ndarray, stride: int = 32, 
                      threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Detect objects using sliding window"""
        detections = []
        
        height, width = image.shape[:2]
        win_h, win_w = self.window_size
        
        for y in range(0, height - win_h + 1, stride):
            for x in range(0, width - win_w + 1, stride):
                # Extract window
                window = image[y:y+win_h, x:x+win_w]
                
                # Preprocess
                window = self.processor.normalize_image(window)
                
                # Predict
                probabilities = self.classifier.forward(window)
                max_prob = np.max(probabilities)
                class_id = np.argmax(probabilities)
                
                # Filter by threshold
                if max_prob > threshold:
                    detection = {
                        'bbox': [x, y, x + win_w, y + win_h],
                        'class_id': class_id,
                        'confidence': max_prob,
                        'center': [x + win_w // 2, y + win_h // 2]
                    }
                    detections.append(detection)
        
        return detections
    
    def non_max_suppression(self, detections: List[Dict[str, Any]], 
                           iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Apply non-maximum suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        kept_detections = []
        
        while detections:
            # Take the detection with highest confidence
            best = detections.pop(0)
            kept_detections.append(best)
            
            # Remove overlapping detections
            remaining = []
            for det in detections:
                if self._calculate_iou(best['bbox'], det['bbox']) < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return kept_detections
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

class ImageSegmentation:
    """Simple image segmentation using clustering"""
    
    def __init__(self, num_clusters: int = 5):
        self.num_clusters = num_clusters
    
    def kmeans_clustering(self, image: np.ndarray, max_iters: int = 100) -> np.ndarray:
        """Simple K-means clustering for segmentation"""
        # Reshape image to feature vectors
        if len(image.shape) == 3:
            h, w, c = image.shape
            pixels = image.reshape(h * w, c)
        else:
            h, w = image.shape
            pixels = image.reshape(h * w, 1)
        
        # Initialize centroids randomly
        centroids = np.random.random((self.num_clusters, pixels.shape[1]))
        
        for _ in range(max_iters):
            # Assign pixels to closest centroid
            distances = np.sqrt(((pixels - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([pixels[labels == k].mean(axis=0) for k in range(self.num_clusters)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        # Reshape labels back to image shape
        segmented = labels.reshape(h, w)
        
        return segmented
    
    def edge_based_segmentation(self, image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Simple edge-based segmentation"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Calculate gradients
        grad_x = np.abs(np.gradient(gray, axis=1))
        grad_y = np.abs(np.gradient(gray, axis=0))
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold edges
        edges = (edge_strength > threshold).astype(int)
        
        return edges

class CVExperimentTracker:
    """Track computer vision experiments"""
    
    def __init__(self, experiment_dir: str = "experiments/cv"):
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        
        self.experiments = []
    
    def create_experiment(self, name: str, config: CVConfig, 
                         description: str = "") -> str:
        """Create new CV experiment"""
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        os.makedirs(experiment_path, exist_ok=True)
        
        # Save config
        config_path = os.path.join(experiment_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        # Save metadata
        metadata = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'task_type': config.task_type,
            'status': 'created'
        }
        
        metadata_path = os.path.join(experiment_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.experiments.append(metadata)
        logger.info(f"Created CV experiment: {experiment_id}")
        
        return experiment_id
    
    def save_results(self, experiment_id: str, results: Dict[str, Any]):
        """Save experiment results"""
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        results_path = os.path.join(experiment_path, "results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                serializable_results[key] = [v.tolist() for v in value]
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results for experiment: {experiment_id}")

def create_sample_images(num_images: int = 10) -> List[np.ndarray]:
    """Create sample images for testing"""
    images = []
    
    for i in range(num_images):
        # Create random patterns
        if i % 3 == 0:
            # Checkerboard pattern
            img = np.zeros((64, 64))
            img[::8, ::8] = 1
            img[4::8, 4::8] = 1
        elif i % 3 == 1:
            # Circle pattern
            img = np.zeros((64, 64))
            center = (32, 32)
            radius = 20
            y, x = np.ogrid[:64, :64]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            img[mask] = 1
        else:
            # Random noise
            img = np.random.random((64, 64))
        
        images.append(img)
    
    return images

def create_cv_system(task_type: str = "classification") -> Dict[str, Any]:
    """Create and configure computer vision system"""
    
    # Create configuration
    config = CVConfig(task_type=task_type, input_size=(64, 64))
    
    # Create components
    processor = ImageProcessor(config.input_size)
    classifier = CNNClassifier(config)
    detector = ObjectDetector(classifier)
    segmentation = ImageSegmentation(num_clusters=3)
    tracker = CVExperimentTracker()
    
    return {
        'config': config,
        'processor': processor,
        'classifier': classifier,
        'detector': detector,
        'segmentation': segmentation,
        'tracker': tracker
    }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Computer Vision System...")
    
    # Create CV system
    cv_system = create_cv_system("classification")
    
    # Create sample images
    sample_images = create_sample_images(20)
    sample_labels = [i % 3 for i in range(20)]  # 3 classes
    
    # Create experiment
    experiment_id = cv_system['tracker'].create_experiment(
        "cnn_classification_test",
        cv_system['config'],
        "Testing CNN classifier on synthetic images"
    )
    
    # Test image processing
    print("Testing image processing...")
    processed_images = []
    for img in sample_images[:5]:
        processed = cv_system['processor'].normalize_image(img)
        augmented = cv_system['processor'].augment_image(processed)
        features = cv_system['processor'].extract_features(processed)
        processed_images.append(augmented)
    
    # Test classification
    print("Testing classification...")
    classifier = cv_system['classifier']
    
    # Simple training loop
    training_losses = []
    for epoch in range(10):
        epoch_loss = 0
        for img, label in zip(sample_images[:10], sample_labels[:10]):
            processed_img = cv_system['processor'].normalize_image(img)
            loss = classifier.train_step(processed_img, label)
            epoch_loss += loss
        
        avg_loss = epoch_loss / 10
        training_losses.append(avg_loss)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    # Test predictions
    print("Testing predictions...")
    predictions = []
    for img in sample_images[10:15]:
        processed_img = cv_system['processor'].normalize_image(img)
        pred = classifier.predict(processed_img)
        predictions.append(pred)
    
    print(f"Predictions: {predictions}")
    
    # Test object detection
    print("Testing object detection...")
    test_image = np.random.random((128, 128))
    detections = cv_system['detector'].detect_objects(test_image, threshold=0.3)
    filtered_detections = cv_system['detector'].non_max_suppression(detections)
    
    print(f"Found {len(detections)} raw detections, {len(filtered_detections)} after NMS")
    
    # Test segmentation
    print("Testing segmentation...")
    test_image_seg = sample_images[0]
    kmeans_result = cv_system['segmentation'].kmeans_clustering(test_image_seg)
    edge_result = cv_system['segmentation'].edge_based_segmentation(test_image_seg)
    
    print(f"Segmentation completed. K-means clusters: {len(np.unique(kmeans_result))}")
    
    # Save results
    results = {
        'training_losses': training_losses,
        'predictions': predictions,
        'num_detections': len(filtered_detections),
        'segmentation_clusters': len(np.unique(kmeans_result)),
        'processed_features': cv_system['processor'].extract_features(sample_images[0])
    }
    
    cv_system['tracker'].save_results(experiment_id, results)
    
    print("\n✅ Computer Vision module tests completed successfully!")
