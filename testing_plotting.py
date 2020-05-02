import matplotlib.pyplot as plt

def make_figure(start, label):
    plt.figure(figsize=(12, 8))
    x = list(range(11))
    y = list(range(start,start+11))
    print(y)
    plt.plot(x, y, label=label)
    plt.plot([1,2,3,4,5], [6,7,8,9,10], label='testing')
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Number of topics")
    plt.ylabel(" coherence")
    plt.legend(title='Models', loc='best')
    file_path = 'reports/figures/' + 'testing' + label + '.png'
    plt.savefig(file_path, bbox_inches='tight')

make_figure(1, 'One1')
make_figure(19, 'Nineteen1')