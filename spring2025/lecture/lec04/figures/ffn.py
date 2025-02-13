import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_trapezoid(bottom_length, top_length, height):
    """
    Create coordinates for a trapezoid centered at the origin,
    with the bottom edge at y=0 and the top edge at y=height.
    """
    return np.array([
        [-bottom_length/2, 0],
        [bottom_length/2, 0],
        [top_length/2, height],
        [-top_length/2, height]
    ])

def rotate_coords(coords, angle_deg):
    """Rotate coordinates by a given angle in degrees."""
    theta = np.radians(angle_deg)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return coords @ rot_matrix.T

def translate_coords(coords, dx, dy):
    """Translate coordinates by dx and dy."""
    return coords + np.array([dx, dy])

# Parameters for the trapezoid: bottom edge is shorter, top edge is longer
bottom_length = 1
top_length = 2
height = 1

# Create the base trapezoid
trapezoid = create_trapezoid(bottom_length, top_length, height)

# Create the expansion trapezoid: rotate by -90° so that the longer edge is on the right.
exp_coords = rotate_coords(trapezoid, -90)
exp_coords = translate_coords(exp_coords, -2, 0)

# Create the projection trapezoid: rotate by 90° so that the longer edge is on the left.
proj_coords = rotate_coords(trapezoid, 90)
proj_coords = translate_coords(proj_coords, 2, 0)

# Set up the plot
fig, ax = plt.subplots(figsize=(10,5))

# Add the trapezoids using patches
exp_trap = patches.Polygon(exp_coords, closed=True, edgecolor='blue', facecolor='lightblue', lw=2)
proj_trap = patches.Polygon(proj_coords, closed=True, edgecolor='green', facecolor='lightgreen', lw=2)
ax.add_patch(exp_trap)
ax.add_patch(proj_trap)

# Draw an arrow between the two trapezoids
arrow = patches.FancyArrowPatch(( -0.5, 0 ), (0.5, 0),
                                 arrowstyle='->', mutation_scale=20, lw=2)
ax.add_patch(arrow)
ax.text(0, 0.2, "$d_{ff}$", fontsize=12, ha="center", va="bottom", color='purple')

arrow = patches.FancyArrowPatch(( -3.5, 0 ), (-2.5, 0),
                                 arrowstyle='->', mutation_scale=20, lw=2)
ax.add_patch(arrow)
ax.text(-3, 0.2, "$d_{model}$", fontsize=12, ha="center", va="bottom", color='purple')

arrow = patches.FancyArrowPatch(( 2.5, 0 ), (3.5, 0),
                                 arrowstyle='->', mutation_scale=20, lw=2)
ax.add_patch(arrow)
ax.text(3, 0.2, "$d_{model}$", fontsize=12, ha="center", va="bottom", color='purple')

# Rotated labels inside the trapezoids.
# For the expansion trapezoid, rotate text by -90°.
ax.text(-1.5, 0, "Expand", fontsize=12, ha="center", va="center", rotation=90, color='blue')
# For the projection trapezoid, rotate text by 90°.
ax.text(1.5, 0, "Project", fontsize=12, ha="center", va="center", rotation=90, color='green')

# Adjust plot limits and aspect ratio
ax.set_xlim(-5, 5)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('ffn.pdf', bbox_inches='tight', pad_inches=0)

