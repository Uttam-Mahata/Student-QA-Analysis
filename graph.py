import matplotlib.pyplot as plt
import numpy as np

def generate_bar_graph(data, txt = "100%"):
    predicted = [x[0] for x in data]
    actual = [x[1] for x in data]
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(20, 10))
    x = range(len(predicted))

    rects1 = ax.bar(x, predicted, width, label='Predicted')
    rects2 = ax.bar([i + width for i in x], actual, width, label='Actual')

    ax.set_ylabel('Values')
    ax.set_title('Predicted vs Actual')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels([str(i) for i in range(1,len(predicted)+1)])
    ax.legend()
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.9)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    plt.annotate(f"Error - {txt}%", xy=(0.4, 0.035), xycoords='figure fraction', fontsize=14)
    plt.show()
    # plt.figure(3)
    # plt.switch_backend('TkAgg') #default on my system
    # print '#3 Backend:',plt.get_backend()
    # plt.plot([1,2,6,4])
    # figManager = plt.get_current_fig_manager()
    # figManager.full_screen_toggle()
    # print(help(plt.savefig))
    # plt.savefig(f'output\\bar_graph{thres}.pdf',bbox_inches='tight',dpi=1000)
    # plt.savefig()
    # plt.show()


if __name__ == "__main__":
    # Example usage:
    # data = [[10, 8], [15, 12], [20, 18], [25, 22]]  # Example data
    # data = {1: [63, 56], 2: [72, 66], 3: [69, 62], 4: [68, 66], 5: [56, 55], 6: [57, 61], 7: [73, 68], 8: [72, 69], 9: [64, 56], 10: [56, 55], 11: [72, 68], 12: [72, 67], 13: [74, 77], 14: [59, 61], 15: [54, 49], 16: [63, 59], 17: [70, 63], 18: [77, 77], 19: [68, 66], 20: [70, 68], 21: [61, 61], 22: [55, 58], 23: [62, 61], 24: [60, 59], 25: [61, 60], 26: [70, 64], 27: [79, 74], 28: [71, 69], 29: [58, 60], 30: [60, 65], 31: [52, 51], 32: [63, 61], 33: [59, 61], 34: [57, 54], 35: [69, 68], 36: [66, 64], 37: [56, 56], 38: [59, 59], 39: [72, 75], 40: [81, 81]}
    data = {1: [63, 56], 2: [72, 66], 3: [69, 62], 4: [68, 66], 5: [56, 55], 6: [57, 61], 7: [73, 68], 8: [72, 69], 9: [64, 56], 10: [56, 55], 11: [72, 68], 12: [72, 67], 13: [74, 77], 14: [59, 61], 15: [54, 49], 16: [63, 59], 17: [70, 63], 18: [77, 77], 19: [68, 66], 20: [70, 68], 21: [61, 61], 22: [55, 58], 23: [62, 61], 24: [60, 59], 25: [61, 60], 26: [70, 64], 27: [79, 74], 28: [71, 69], 29: [58, 60], 30: [60, 65], 31: [52, 51], 32: [63, 61], 33: [59, 61], 34: [57, 54], 35: [69, 68], 36: [66, 64], 37: [56, 56], 38: [59, 59], 39: [72, 75], 40: [81, 81]}
    data_ner = {1: [75, 73], 2: [85, 81], 3: [82, 81], 4: [82, 80], 5: [68, 67], 6: [77, 74], 7: [87, 84], 8: [88, 85], 9: [74, 73], 10: [78, 76], 11: [88, 85], 12: [87, 83], 13: [82, 80], 14: [65, 65], 15: [79, 78], 16: [82, 80], 17: [83, 81], 18: [87, 85], 19: [78, 76], 20: [84, 81], 21: [74, 73], 22: [71, 69], 23: [80, 79], 24: [65, 64], 25: [68, 65], 26: [82, 81], 27: [92, 90], 28: [88, 85], 29: [74, 73], 30: [76, 76], 31: [67, 65], 32: [75, 74], 33: [80, 79], 34: [71, 71], 35: [85, 82], 36: [82, 77], 37: [71, 70], 38: [71, 70], 39: [89, 86], 40: [90, 89]}

    acc1 = 0
    acc2 = 0
    for p,a in data_ner.values():
        acc1 += abs(a-p)/a
    for p,a in data.values():
        acc2 += abs(a-p)/a
        # acc1 /= a
    # acc1 = acc1**0.5
    acc1 = acc1/(len(data_ner))*100
    acc2 = acc2/(len(data))*100
    # print((1-acc1/acc2)*100)
    print("ner - ",acc1)
    print("full - ",acc2)

    generate_bar_graph(list(data.values()), round(acc1,2))

    # Test the function with some data
    # data = [[1, 2], [2, 3], [3, 2], [4, 5], [5, 6]]
# plot_bar_graph(list(data.values()))


