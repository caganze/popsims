import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

norm=mpl.colors.LogNorm()
def plot_annotated_heatmap(ax, data, gridpoints, columns, cmap='viridis', 
                           annotate=False,  textsize=14, alpha=0.1,norm=norm ):
    """Plots an annotated heatmap.
    Args:
        ax (matplotlib.axis): Axis to plot on.
        data (pandas.DataFrame): Data to plot.
        gridpoints (int): Number of grid points to use.
        columns (list): List of length 3 containing the names of the columns to use as the x, y, and z values of the heatmap.
        cmap (str): Colormap to use (default 'viridis').
        annotate (bool): Whether to annotate the heatmap with the median or count of z values (default False).
        vmin (float): Minimum value of the color range (default 0.0).
        vmax (float): Maximum value of the color range (default 1.0).
        textsize (int): Font size for annotations (default 14).
        alpha (float): Transparency of the heatmap (default 0.1).

    Returns:
        matplotlib.axis: The axis with the plotted heatmap.

    """
    #plot an annotated heatmap
    data= data.dropna()
    xcol, ycol, zcol= columns
    step1= np.ptp(data[xcol])/gridpoints
    step2= np.ptp(data[ycol])/gridpoints
    

    #print (step1)
    
    xgrid= np.linspace(data[xcol].min(), data[xcol].max(), gridpoints)
    ygrid= np.linspace(data[ycol].min(), data[ycol].max(), gridpoints)
    
    
    mask = np.zeros((len(xgrid), len(ygrid)))
    values = np.zeros((len(xgrid), len(ygrid)))
    #for annotation
    for i in range(len(xgrid)):
        #loop over matrix
        for j in range(len(ygrid)):
            if (i == len(xgrid)-1) | (j == len(ygrid)-1) :
                pass
            else:
                maskx= np.logical_and(data[xcol] > xgrid[i], data[xcol] <= xgrid[i]+step1)
                masky=np.logical_and(data[ycol] > ygrid[j], data[ycol] <=ygrid[j]+step2)
                zmedian= np.nanmean(data[zcol][np.logical_and(maskx, masky)])
                lenz= len(data[np.logical_and.reduce([maskx, masky])])

                if lenz == 0:
                    values[j][i] = np.nan
                    mask[j][i] = 1
                else:
                    values[j][i] = zmedian
                    if annotate == 'third_value':
                        ax.text(xgrid[i]+step1/2., ygrid[j]+step2/2., f'{zmedian:.0f}',
                                 ha='center', va='center', fontsize=textsize, color='#111111')
                    if annotate== 'number':
                        ax.text(xgrid[i]+step1/2., ygrid[j]+step2/2., f'{lenz:.0f}',
                                 ha='center', va='center', fontsize=textsize, color='#111111')
                
    values2 = np.ma.array(values, mask=mask)
    cax = ax.pcolormesh(xgrid, ygrid, values2,  cmap=cmap, alpha=alpha, norm=norm)
    #plt.axis('tight')
    ymin, ymax = plt.ylim()

    ax.minorticks_on()

    ax.set_ylim(ymax, ymin)
    return 

