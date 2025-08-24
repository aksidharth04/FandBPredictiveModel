import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_fb_process_flow():
    """Create a visual flow diagram of the F&B manufacturing process"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define colors
    colors = {
        'raw_materials': '#FFE4B5',
        'mixing': '#98FB98',
        'fermentation': '#87CEEB',
        'baking': '#FFB6C1',
        'quality': '#DDA0DD',
        'arrow': '#2F4F4F'
    }
    
    # Process steps
    steps = [
        {'name': 'Raw Materials\nPreparation', 'pos': (1, 6), 'size': (1.5, 1), 'color': colors['raw_materials']},
        {'name': 'Mixing Phase', 'pos': (3, 6), 'size': (1.5, 1), 'color': colors['mixing']},
        {'name': 'Fermentation\nPhase', 'pos': (5, 6), 'size': (1.5, 1), 'color': colors['fermentation']},
        {'name': 'Baking Phase', 'pos': (7, 6), 'size': (1.5, 1), 'color': colors['baking']},
        {'name': 'Quality\nAssessment', 'pos': (9, 6), 'size': (1.5, 1), 'color': colors['quality']}
    ]
    
    # Parameters for each step
    parameters = [
        {'step': 0, 'params': ['Flour (kg)', 'Sugar (kg)', 'Yeast (kg)', 'Salt (kg)', 'Water'], 'pos': (1, 4.5)},
        {'step': 1, 'params': ['Mixer Speed (RPM)', 'Mixing Temp (C)', 'Mixing Time'], 'pos': (3, 4.5)},
        {'step': 2, 'params': ['Fermentation Temp (C)', 'Time', 'Humidity'], 'pos': (5, 4.5)},
        {'step': 3, 'params': ['Oven Temp (C)', 'Oven Humidity (%)', 'Baking Time'], 'pos': (7, 4.5)},
        {'step': 4, 'params': ['Final Weight (kg)', 'Quality Score (%)'], 'pos': (9, 4.5)}
    ]
    
    # Draw process steps
    for step in steps:
        box = FancyBboxPatch(
            (step['pos'][0] - step['size'][0]/2, step['pos'][1] - step['size'][1]/2),
            step['size'][0], step['size'][1],
            boxstyle="round,pad=0.1",
            facecolor=step['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(step['pos'][0], step['pos'][1], step['name'], 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows between steps
    for i in range(len(steps) - 1):
        arrow = ConnectionPatch(
            (steps[i]['pos'][0] + steps[i]['size'][0]/2, steps[i]['pos'][1]),
            (steps[i+1]['pos'][0] - steps[i+1]['size'][0]/2, steps[i+1]['pos'][1]),
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=20, fc=colors['arrow'], linewidth=2
        )
        ax.add_patch(arrow)
    
    # Draw parameters
    for param_group in parameters:
        y_offset = 0
        for param in param_group['params']:
            ax.text(param_group['pos'][0], param_group['pos'][1] - y_offset, 
                   f"• {param}", ha='center', va='top', fontsize=9)
            y_offset += 0.3
    
    # Add title and subtitle
    ax.text(5, 7.5, 'F&B Manufacturing Process Flow', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(5, 7.2, 'Baked Goods Production with LSTM Quality Prediction', 
            ha='center', va='center', fontsize=12, style='italic')
    
    # Add LSTM prediction indicator
    ax.text(5, 3.5, 'LSTM Model Predicts Quality Score (%)', 
            ha='center', va='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.7))
    
    # Add performance metrics
    metrics_text = """Model Performance:
• MAE: 5.50%
• RMSE: 6.76%
• 86% within 10% error
• 51% within 5% error"""
    
    ax.text(5, 2.5, metrics_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('fb_process_flow.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_lstm_architecture():
    """Create a visual representation of the LSTM architecture"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # LSTM layers
    layers = [
        {'name': 'Input\n(10 features × 10 timesteps)', 'pos': (1, 4), 'size': (2, 1.5), 'color': '#E6F3FF'},
        {'name': 'LSTM Layer\n(50 units)', 'pos': (4, 4), 'size': (2, 1.5), 'color': '#FFE6E6'},
        {'name': 'Dropout\n(0.2)', 'pos': (7, 4), 'size': (1.5, 1.5), 'color': '#E6FFE6'},
        {'name': 'Dense Layer\n(2 outputs)', 'pos': (10, 4), 'size': (1.5, 1.5), 'color': '#FFFFE6'}
    ]
    
    # Draw layers
    for layer in layers:
        box = FancyBboxPatch(
            (layer['pos'][0] - layer['size'][0]/2, layer['pos'][1] - layer['size'][1]/2),
            layer['size'][0], layer['size'][1],
            boxstyle="round,pad=0.1",
            facecolor=layer['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(layer['pos'][0], layer['pos'][1], layer['name'], 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    for i in range(len(layers) - 1):
        arrow = ConnectionPatch(
            (layers[i]['pos'][0] + layers[i]['size'][0]/2, layers[i]['pos'][1]),
            (layers[i+1]['pos'][0] - layers[i+1]['size'][0]/2, layers[i+1]['pos'][1]),
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=20, fc='black', linewidth=2
        )
        ax.add_patch(arrow)
    
    # Add input features
    features = ['Flour', 'Sugar', 'Yeast', 'Water Temp', 'Salt', 
               'Mixer Speed', 'Mixing Temp', 'Fermentation Temp', 
               'Oven Temp', 'Oven Humidity']
    
    for i, feature in enumerate(features):
        ax.text(1, 6.5 - i*0.3, f"• {feature}", ha='left', va='center', fontsize=8)
    
    # Add output
    ax.text(10, 6, 'Output:\nQuality Score (%)', ha='center', va='center', 
            fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Add title
    ax.text(6, 7.5, 'LSTM Architecture for F&B Quality Prediction', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add model info
    info_text = """Model Details:
• Total Parameters: 12,302
• Time Steps: 10
• Features: 10 process variables
• Output: Quality Score prediction"""
    
    ax.text(6, 1.5, info_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('lstm_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating F&B Process Flow Diagram...")
    create_fb_process_flow()
    
    print("Creating LSTM Architecture Diagram...")
    create_lstm_architecture()
    
    print("Visualizations saved as:")
    print("- fb_process_flow.png")
    print("- lstm_architecture.png")
