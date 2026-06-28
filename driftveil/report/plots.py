import matplotlib.pyplot as plt
import numpy as np
from typing import Any

def plot_drifts(report: Any) -> plt.Figure:
    """Generate a Matplotlib figure summarizing the data drift checks."""
    results = report.results
    
    if not results:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No checks executed", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig
        
    # Categorize results
    labels = []
    statuses = []
    colors = []
    
    for r in results:
        if r.target is None:
            lbl = f"dataset: {r.contract}"
        elif isinstance(r.target, tuple):
            lbl = f"{r.target[0]}/{r.target[1]}: {r.contract}"
        else:
            lbl = f"{r.target}: {r.contract}"
            
        labels.append(lbl)
        
        if r.passed:
            statuses.append(1.0) # Passed
            colors.append("#10b981") # Green
        else:
            if r.severity == "error":
                statuses.append(-1.0) # Failed Error
                colors.append("#ef4444") # Red
            elif r.severity == "warning":
                statuses.append(-0.5) # Failed Warning
                colors.append("#f59e0b") # Orange
            else:
                statuses.append(-0.2) # Failed Info
                colors.append("#3b82f6") # Blue
                
    # Determine figure height dynamically based on number of checks
    fig_height = max(4, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, statuses, align="center", color=colors, height=0.6)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.invert_yaxis() # Top-down layout
    
    # Customize axes
    ax.set_xlim(-1.2, 1.2)
    ax.axvline(0, color="#475569", linewidth=1.5, linestyle="--")
    
    # Hide X ticks and labels, replace with custom legend/annotations
    ax.get_xaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("#475569")
    
    # Title
    ax.set_title("Driftveil Contract Enforcement Summary", fontsize=14, fontweight="bold", pad=20, color="#1e293b")
    
    # Add status labels to bars
    for bar, r in zip(bars, results):
        width = bar.get_width()
        if r.passed:
            ax.text(0.1, bar.get_y() + bar.get_height()/2, "PASSED",
                    va="center", ha="left", color="#10b981", fontsize=9, fontweight="bold")
        else:
            txt = r.severity.upper()
            ax.text(-0.1, bar.get_y() + bar.get_height()/2, txt,
                    va="center", ha="right", color=bar.get_facecolor(), fontsize=9, fontweight="bold")
                    
    plt.tight_layout()
    return fig
