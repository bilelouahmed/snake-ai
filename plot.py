import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
plt.ion()

def plot(gamma:float, scores:list, mean_scores:list=None, epsilons:list=None):
    ax1.clear()
    ax2.clear()
    
    ax1.set_title(f"Snake Training - Gamma : {gamma}")
    ax1.set_xlabel('Number of games')
    ax1.set_ylabel('Score')
    ax1.plot(scores, label='Scores', color='blue')
    ax1.plot(mean_scores, label='Mean Scores', color='orange')
    ax1.set_ylim(ymin=0)
    ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
    ax1.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    if epsilons:
        ax2.set_ylabel('Epsilon', )
        ax2.plot(epsilons, label='Epsilons', color='green')
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(axis='y')

        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.2f}%"))

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.draw()
    plt.pause(0.2)