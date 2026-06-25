import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib
import matplotlib as mpl
import matplotlib.patches as patches  
from numba import njit

@njit
def generate_seed():
    return random.getrandbits(32)

def __pop_random(lst):
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)

def map_pairs(length):
    if length %2 == 1:
        length -=1
    lst = []
    for i in range(length):
        lst.append(i)
    pairs = []
    while lst:
        rand1 = __pop_random(lst)
        rand2 = __pop_random(lst)
        pair = rand1, rand2
        pairs.append(pair)
    
    return pairs

def print_error_rate(array1,array2,signal_bits = []):
    counter = 0
    for i in range(len(array1)):
        if array1[i] != array2[i]:
            counter +=1
    print("faulty bits: " + str(counter))
    print("error rate: " + str(counter/len(array1)))



def make_heatmap(array,path):
    count_photons = 0
    for i in range(len(array)):
        count_photons+= array[i][0] + array[i][1]
    print("measured photons: " + str(count_photons))



    heatmap = np.zeros((6,6))
    for i in range(len(array)):
        if array[i][0] <= 5 and array[i][1] <= 5 :

            heatmap[int(array[i][0])][int(array[i][1])] += 1




    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)

    # Show all ticks and label them with the respective list entries
    #ax.set_xticks(range(len(farmers)), labels=farmers,
    #            rotation=45, ha="right", rotation_mode="anchor")
    #ax.set_yticks(range(len(vegetables)), labels=vegetables)

    # Loop over data dimensions and create text annotations.
    for i in range(len(heatmap[0])):
        for j in range(len(heatmap)):
            text = ax.text(j, i, heatmap[i, j],
                        ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax)


    ax.set_title("detecor clicks")
    plt.xlabel("Photons on side D0")
    plt.ylabel("Photons on side D1")
    fig.tight_layout()
    plt.savefig(path)

    return count_photons
    #plt.show()

def make_heatmap_array(array,path,n):
    array = array[0:n,0:n]
    fig, ax = plt.subplots()
    im = ax.imshow(array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(
                j, i,
                f"{array[i, j]:.2f}",
                ha="center",
                va="center",
                color="white"
            )

    
    rect = patches.Rectangle(
        xy=(0.5, -0.485),      # Start at x=0.5 (left edge of col 1), y=-0.5 (top edge of row 0)
        width=3,             # Cover 4 columns (1, 2, 3, 4)
        height=0.985,            # Cover 1 row (0)
        linewidth=3,         # Thickness of the border
        edgecolor='lime',    # Bright green color ('g' works too, but 'lime' pops more on heatmaps)
        facecolor='lime',     # Keep the inside transparent
        alpha=0.2
    )
    ax.add_patch(rect)
    rect2 = patches.Rectangle(
        xy=(-0.485, 0.5),      # Start at x=0.5 (left edge of col 1), y=-0.5 (top edge of row 0)
        width=0.985,             # Cover 4 columns (1, 2, 3, 4)
        height=3,            # Cover 1 row (0)
        linewidth=3,         # Thickness of the border
        edgecolor='lime',    # Bright green color ('g' works too, but 'lime' pops more on heatmaps)
        facecolor='lime',     # Keep the inside transparent
        alpha=0.2
    )
    ax.add_patch(rect2)
    rect3 = patches.Rectangle(
        xy=(0.515, 0.515),      # Start at x=0.5 (left edge of col 1), y=-0.5 (top edge of row 0)
        width=2.985,             # Cover 4 columns (1, 2, 3, 4)
        height=2.985,            # Cover 1 row (0)
        linewidth=3,         # Thickness of the border
        edgecolor='red',    # Bright green color ('g' works too, but 'lime' pops more on heatmaps)
        facecolor='red',     # Keep the inside transparent
        alpha=0.2
    )
    ax.add_patch(rect3)

    #fig.colorbar(im, ax=ax)

    ax.set_title("detecor clicks")
    plt.xlabel("Photons on side D0")
    plt.ylabel("Photons on side D1")
    fig.tight_layout()
    plt.savefig(path)