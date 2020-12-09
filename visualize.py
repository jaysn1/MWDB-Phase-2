import numpy as np

color_dict = {"aqua": "#00ffff","azure": "#f0ffff","beige": "#f5f5dc","black": "#000000","blue": "#0000ff","brown": "#a52a2a","cyan": "#00ffff","darkblue": "#00008b","darkcyan": "#008b8b","darkgrey": "#a9a9a9","darkgreen": "#006400","darkkhaki": "#bdb76b","darkmagenta": "#8b008b","darkolivegreen": "#556b2f","darkorange": "#ff8c00","darkorchid": "#9932cc","darkred": "#8b0000","darksalmon": "#e9967a","darkviolet": "#9400d3","fuchsia": "#ff00ff","gold": "#ffd700","green": "#008000","indigo": "#4b0082","khaki": "#f0e68c","lightblue": "#add8e6","lightcyan": "#e0ffff","lightgreen": "#90ee90","lightgrey": "#d3d3d3","lightpink": "#ffb6c1","lightyellow": "#ffffe0","lime": "#00ff00","magenta": "#ff00ff","maroon": "#800000","navy": "#000080","olive": "#808000","orange": "#ffa500","pink": "#ffc0cb","purple": "#800080","violet": "#800080","red": "#ff0000","silver": "#c0c0c0","white": "#ffffff","yellow": "#ffff00"}
clrs = ["black","blue","cyan","green","azure","beige","darkblue","darkcyan","darkgrey","darkgreen","darkkhaki","darkmagenta","darkolivegreen","darkorange","darkorchid","darkred","darksalmon","darkviolet","khaki","gold"]
clrs = list(map(lambda x: color_dict[x], clrs))
    
def visualize(data, plt, N=None, resolution=4, file=''):
    """
    Given the data matrix
    """
    fig = plt.figure 
    plt.set_title(file)
    # fig.show()
    fig.canvas.draw()
    if N == None:
        N = data.shape[0]
    for i in range(len(data[:N])):
        xs, ys = range(len(data[i])), data[i]
        plt.plot(xs, ys, c=clrs[i]) #clrs[i])
        fig.canvas.draw()

def main():
    from Phase1.helper import min_max_scaler, load_data
    import matplotlib.pyplot as plt

    gestures = ['1', '1_1', '1_2']
    data = "C:/Users/monil/Desktop/CSE 515 - MWDB/Project/phase3_git/data"
    resolution=4

    for gesture in gestures:
        (x, y, z, w) = (np.array(load_data(f"{data}/X/{gesture}.csv")),
                    np.array(load_data(f"{data}/Y/{gesture}.csv")), 
                    np.array(load_data(f"{data}/Z/{gesture}.csv")), 
                    np.array(load_data(f"{data}/W/{gesture}.csv")))

        plt.figure(figsize=(18,10))
        visualize(x, plt.subplot(2,2,1), None, resolution, f'{gesture}-X')
        visualize(y, plt.subplot(2,2,2), None, resolution, f'{gesture}-Y')
        visualize(z, plt.subplot(2,2,3), None, resolution, f'{gesture}-Z')
        visualize(w, plt.subplot(2,2,4), None, resolution, f'{gesture}-W')
        plt.show(block=False)
    plt.show()
    plt.close("all")
if __name__=="__main__":
    main()