"""
TSM Two-Stream Fusion Network for RGB-D Gesture Recognition

This module implements a modular TSM-based network that supports various configurations
for ablation studies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalShiftModule(nn.Module):
    """Temporal Shift Module (TSM) for efficient temporal modeling"""
    
    def __init__(self, n_segment=8, n_div=8, fold_div=3):
        super(TemporalShiftModule, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div
        self.n_div = n_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        
        # Shift part of channels forward and backward in time
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]  # shift right
        out[:, :, 2*fold:] = x[:, :, 2*fold:]  # no shift
        
        return out.view(nt, c, h, w)


class ResNetBlock(nn.Module):
    """Basic ResNet block with TSM"""
    
    def __init__(self, in_channels, out_channels, stride=1, use_tsm=True, n_segment=8):
        super(ResNetBlock, self).__init__()
        self.use_tsm = use_tsm
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if use_tsm:
            self.tsm = TemporalShiftModule(n_segment=n_segment)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        if self.use_tsm:
            x = self.tsm(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TransformerFusionModule(nn.Module):
    """Transformer-based feature fusion module"""
    
    def __init__(self, d_model=512, nhead=8, num_layers=2, dim_feedforward=2048):
        super(TransformerFusionModule, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, rgb_features, depth_features):
        # Concatenate RGB and depth features along sequence dimension
        combined = torch.cat([rgb_features, depth_features], dim=1)
        fused = self.transformer(combined)
        return fused


class MultiScaleTemporalPooling(nn.Module):
    """Multi-scale temporal pooling with different pooling sizes"""
    
    def __init__(self, n_segment=8):
        super(MultiScaleTemporalPooling, self).__init__()
        self.n_segment = n_segment
        
    def forward(self, x):
        # x shape: (batch, channels, time, h, w) or (batch, channels)
        if len(x.shape) == 2:
            # Already pooled spatially, add temporal dimension
            n, c = x.shape
            x = x.view(n // self.n_segment, self.n_segment, c)
            
            # Apply different temporal pooling strategies
            max_pool = torch.max(x, dim=1)[0]
            avg_pool = torch.mean(x, dim=1)
            
            # Concatenate different pooling results
            out = torch.cat([max_pool, avg_pool], dim=1)
            return out
        else:
            # Apply pooling on temporal dimension
            return torch.mean(x, dim=2)


class TSMTwoStreamFusionNetwork(nn.Module):
    """
    Modular TSM Two-Stream Fusion Network supporting various configurations
    for ablation studies.
    """
    
    def __init__(self, num_classes=83, n_segment=8, 
                 use_rgb=True, use_depth=True, 
                 use_transformer=True, pooling_type='multi_scale',
                 dropout_rate=0.5):
        super(TSMTwoStreamFusionNetwork, self).__init__()
        
        self.num_classes = num_classes
        self.n_segment = n_segment
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_transformer = use_transformer
        self.pooling_type = pooling_type
        
        # RGB stream
        if use_rgb:
            self.rgb_stream = self._make_stream('rgb')
        
        # Depth stream
        if use_depth:
            self.depth_stream = self._make_stream('depth')
        
        # Feature dimension after backbone
        feature_dim = 512
        
        # Transformer fusion
        if use_transformer and use_rgb and use_depth:
            self.transformer_fusion = TransformerFusionModule(d_model=feature_dim)
            fusion_dim = feature_dim * 2  # Combined RGB and depth after transformer
        else:
            fusion_dim = feature_dim * (int(use_rgb) + int(use_depth))
        
        # Temporal pooling
        if pooling_type == 'multi_scale':
            self.temporal_pooling = MultiScaleTemporalPooling(n_segment=n_segment)
            # Multi-scale pooling concatenates max and avg pooling
            classifier_input_dim = fusion_dim * 2
        else:
            # Simple average pooling
            classifier_input_dim = fusion_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_stream(self, stream_type):
        """Create a feature extraction stream (RGB or Depth)"""
        layers = []
        
        # Input channels: 3 for RGB, 1 for depth
        in_channels = 3 if stream_type == 'rgb' else 1
        
        # Initial conv layer
        layers.append(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # ResNet blocks with TSM
        layers.append(ResNetBlock(64, 128, stride=1, use_tsm=True, n_segment=self.n_segment))
        layers.append(ResNetBlock(128, 256, stride=2, use_tsm=True, n_segment=self.n_segment))
        layers.append(ResNetBlock(256, 512, stride=2, use_tsm=True, n_segment=self.n_segment))
        
        # Global average pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb_input=None, depth_input=None):
        """
        Forward pass
        
        Args:
            rgb_input: RGB frames (batch * n_segment, 3, H, W)
            depth_input: Depth frames (batch * n_segment, 1, H, W)
        
        Returns:
            logits: Classification logits (batch, num_classes)
        """
        features = []
        
        # Extract RGB features
        if self.use_rgb and rgb_input is not None:
            rgb_feat = self.rgb_stream(rgb_input)
            rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)  # Flatten
            features.append(rgb_feat)
        
        # Extract depth features
        if self.use_depth and depth_input is not None:
            depth_feat = self.depth_stream(depth_input)
            depth_feat = depth_feat.view(depth_feat.size(0), -1)  # Flatten
            features.append(depth_feat)
        
        # Fusion
        if self.use_transformer and self.use_rgb and self.use_depth and len(features) == 2:
            # Reshape for transformer: (batch, seq_len, feature_dim)
            batch_size = features[0].size(0) // self.n_segment
            rgb_seq = features[0].view(batch_size, self.n_segment, -1)
            depth_seq = features[1].view(batch_size, self.n_segment, -1)
            
            fused = self.transformer_fusion(rgb_seq, depth_seq)
            # Average over sequence dimension for transformer
            if self.pooling_type == 'simple_avg':
                fused = torch.mean(fused, dim=1)
            else:
                # For multi-scale, keep the sequence dimension
                fused = fused.view(batch_size, -1)  # Flatten sequence
        else:
            # Simple concatenation without transformer
            if len(features) > 0:
                # Concatenate features from different modalities
                # Shape: (batch*n_segment, feature_dim)
                fused = torch.cat(features, dim=1)
                
                # Reshape to (batch, n_segment, feature_dim)
                batch_size = fused.size(0) // self.n_segment
                fused = fused.view(batch_size, self.n_segment, -1)
                
                # Temporal aggregation
                if self.pooling_type == 'simple_avg':
                    fused = torch.mean(fused, dim=1)
                else:
                    # For multi-scale, flatten
                    fused = fused.view(batch_size, -1)
            else:
                raise ValueError("No valid input provided!")
        
        # Multi-scale temporal pooling (if enabled)
        if self.pooling_type == 'multi_scale':
            # fused is already flattened, apply multi-scale pooling
            # Reshape to have temporal dimension
            batch_size = fused.size(0)
            fused_dim = fused.size(1) // self.n_segment
            fused = fused.view(batch_size, self.n_segment, fused_dim)
            
            # Apply different pooling strategies
            max_pool = torch.max(fused, dim=1)[0]
            avg_pool = torch.mean(fused, dim=1)
            
            # Concatenate different pooling results
            fused = torch.cat([max_pool, avg_pool], dim=1)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
    
    def get_parameter_count(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameter_count(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config):
    """
    Factory function to create model based on configuration
    
    Args:
        config: Dictionary with model configuration
            - num_classes: Number of gesture classes
            - n_segment: Number of temporal segments
            - use_rgb: Whether to use RGB stream
            - use_depth: Whether to use depth stream
            - use_transformer: Whether to use transformer fusion
            - pooling_type: 'simple_avg' or 'multi_scale'
    
    Returns:
        model: Configured TSMTwoStreamFusionNetwork instance
    """
    return TSMTwoStreamFusionNetwork(
        num_classes=config.get('num_classes', 83),
        n_segment=config.get('n_segment', 8),
        use_rgb=config.get('use_rgb', True),
        use_depth=config.get('use_depth', True),
        use_transformer=config.get('use_transformer', True),
        pooling_type=config.get('pooling_type', 'multi_scale'),
        dropout_rate=config.get('dropout_rate', 0.5)
    )
