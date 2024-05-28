
import matplotlib.pyplot as plt
import numpy as np
import plot_data
def bar_gi_bi():
    species = ("$(\gamma=0.25, \delta=2$)", "$(\gamma=0.25, \delta=4$)", "$(\gamma=0.5, \delta=2$)", "($\gamma=0.5, \delta=4$)")
    penguin_means = {
        '$\hat{g}^o_i$ ground truth of the number of green tokens': (67.49, 52.75, 122.77, 120.67),
        '$\hat{b}_i$ substitution bound': (71.44, 55.12, 124.75, 121.89),
        '$\hat{g}_i$ watermark threshold': (48.90, 30.31, 99.08, 91.72),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(
            x + offset, measurement, width, 
            label=attribute, 
            alpha=0.8,
            hatch='//',
            # color=['blue', 'red', 'yellow']
        )
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # ax.xticks(size = 16)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Length (mm)')
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, species, size = 14, rotation=-10)
    # plt.xticks(x, ['{}'.format(i) for i in x])
    # ax.legend(loc='upper right', ncols=3)
    legend = ax.legend(loc='upper center', fontsize=15)#, shadow=True, fontsize='x-large'
    # legend.get_frame().set_facecolor('C0')
    ax.set_ylim(0, 180)

    plt.savefig('plot/g_i_b_i.pdf')

def plot_gi_bi(model_name):
    
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(20, 4.5)
    legend=['o-', 'o--', '<-', '<--','>-', '>--','*-', '*--']
    color=['b', 'b', 'r', 'r','y', 'y','g', 'g']
    for idx, (gamma,delta) in enumerate(plot_data.gi_bi[model_name]):
        min_y, max_y=100,0
        for idy, (attribute, measurement) in enumerate(plot_data.gi_bi[model_name][(gamma,delta)].items()):
            
            ax[idx].plot(
                ['4000','10000','20000','40000'],
                measurement, 
                legend[idy],
                color=color[idy],
                label=attribute, 
            )
            # else:
            #     ax[idx].plot(
            #         ['4000','10000','20000','40000'],
            #         measurement, 
            #         color=color[idy],
            #         label=attribute, 
            #     )
            if min_y>=min(measurement):
                min_y=min(measurement)
            if max_y<=max(measurement):
                max_y=max(measurement)
        # ax[idx].set_size_inches(4,6)
        # ax[idx].legend(loc='upper center', ncols=3, fontsize=8)
        if idx==0:
            ax[idx].set_ylabel(model_name, fontsize=18)
        ax[idx].set_xlabel('$\gamma$='+str(gamma)+', $\delta$='+str(delta), fontsize=14)
        ax[idx].set_ylim((min_y-10, max_y+5))
    ax[3].legend(loc='upper center', bbox_to_anchor=(-1.3, 1.5),fancybox=True, shadow=True, ncol=2, fontsize=14)
    plt.subplots_adjust(wspace=0.2)
    plt.savefig('plot/'+model_name+'_g_i_b_i.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

if __name__=='__main__':
    bar_gi_bi()
    # model_name='OPT'
    # plot_gi_bi(model_name)
    # model_name='LLaMA'
    # plot_gi_bi(model_name)