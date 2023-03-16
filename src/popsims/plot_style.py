

def plot_style():
        #import some plotting functions
        import seaborn
        seaborn.set_style("ticks")

        #give an option to set up plotting style

        import matplotlib as mpl 
        #mpl.pyplot.style.use('fivethirtyeight')
        #matplotlib defaults
        mpl.rcParams['grid.color'] = 'k'
        mpl.rcParams['grid.linestyle'] = '--'
        mpl.rcParams['grid.linewidth'] = 0.5
        mpl.rcParams['axes.linewidth'] = 1.5
        mpl.rcParams['figure.figsize'] = [8.0, 6.0]
        mpl.rcParams['figure.dpi'] = 80
        mpl.rcParams['savefig.dpi'] = 100
        mpl.rcParams['font.size'] = 16
        mpl.rcParams['legend.fontsize'] = 'large'
        mpl.rcParams['figure.titlesize'] = 'large'
        mpl.rcParams['xtick.bottom']=True
        mpl.rcParams['xtick.top']=True
        mpl.rcParams['xtick.major.width']=0.5
        mpl.rcParams['xtick.minor.width']=0.5
        mpl.rcParams['ytick.major.width']=0.5
        mpl.rcParams['ytick.minor.width']=0.5
        mpl.rcParams['ytick.right']=True
        mpl.rcParams['ytick.left']=True
        mpl.rcParams['xtick.direction']='out'
        mpl.rcParams['ytick.direction']='out'

        mpl.rcParams['font.serif'] = 'Ubuntu'
        #mpl.rcParams['font.monospace'] = 'Ubuntu Mono'
        mpl.rcParams["mathtext.fontset"] = "dejavuserif"

        mpl.rcParams['axes.labelsize'] = 16
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['legend.fontsize'] = 16
        mpl.rcParams['figure.titlesize'] = 16

        font = {'family' : 'serif',
                'serif':['Verdana'],
                'weight' : 'normal',
                'size'   : 18}

        mpl.rc('font', **font)