"""
Plot configuration for top-tier journal figures (Nature/Science style).
- Sans-serif fonts (Arial/Helvetica)
- Colorblind-safe palette (Wong)
- Vector export (PDF) with embedded fonts
- Journal column widths: 89 mm single, 183 mm double
"""
import matplotlib.pyplot as plt

# Font: Nature requires Arial or Helvetica (sans-serif)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

# Text sizes (Nature: 5-7 pt body, 8 pt bold for panel labels)
plt.rcParams["font.size"] = 7
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6
plt.rcParams["legend.fontsize"] = 6
plt.rcParams["figure.titlesize"] = 8

# Axes: clean style, no top/right spines
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.grid"] = False  # Nature recommends avoiding gridlines

# Lines and ticks
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1

# PDF export: embed fonts (TrueType 42) for editable text
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "path"

# Figure DPI for raster export
FIG_DPI = 300

# Journal column widths (mm to inches)
SINGLE_COL_MM = 89
DOUBLE_COL_MM = 183
MM_TO_INCH = 0.03937


def figsize_single(w_ratio=1.0, h_ratio=0.75):
    """Single column: 89 mm × aspect."""
    w = SINGLE_COL_MM * MM_TO_INCH * w_ratio
    h = w * h_ratio
    return (w, h)


def figsize_double(w_ratio=1.0, h_ratio=0.5):
    """Double column: 183 mm × aspect."""
    w = DOUBLE_COL_MM * MM_TO_INCH * w_ratio
    h = w * h_ratio
    return (w, h)


# Wong colorblind-safe palette (Nature Methods 8, 441)
# Order: blue, orange, sky blue, green, yellow, blue, vermillion, purple
PALETTE_WONG = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue (repeat if needed)
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
]

# Algorithm colors (consistent across all figures)
ALGO_COLORS = {
    "ppo": "#0072B2",
    "PPO": "#0072B2",
    "recurrentppo": "#E69F00",
    "Recurrent PPO": "#E69F00",
    "agri_metarl": "#009E73",
    "Agri-MetaRL": "#009E73",
    "rule_baseline": "#999999",
}


def format_algo_name(name):
    """Convert algo key to display label."""
    m = {
        "ppo": "PPO",
        "recurrentppo": "Recurrent PPO",
        "agri_metarl": "Agri-MetaRL",
        "rule_baseline": "Rule baseline",
    }
    return m.get(str(name).lower(), str(name))
