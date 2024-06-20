import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

def draw_selection_pairs(Ns,std,c):
    x = np.linspace(0, Ns-1, 100)
    y = np.exp(-x**2/(2*std**2))
    plt.plot(x, y, linewidth=2.5,color=c)
    x2 = np.linspace(Ns-1, 10, 100)
    y2 = np.exp(-x2**2/(2*std**2))
    plt.plot(x2, y2, linewidth=2.5,color="black",linestyle='--')
    for xi in range(Ns-1,-2,-1):
        plt.plot(xi, np.exp(-xi**2/(2*std**2)), 'o',color=c, markersize=10)
    plt.plot(-1, np.exp(-1**2/(2*std**2)), 'o',color=c, markersize=10, label=r'$(N_s,\sigma)=(%d,%g)$'%(Ns,std))
  
def return_x_for_axis(Ns,std):
    x=[]
    y=[]
    for xi in range(Ns-1,-1,-1):
        yi=np.exp(-xi**2/(2*std**2))
        x.append(xi)
        y.append(round(yi,2))
    return x,y

    

def plot_gaussian(std):
    x = np.linspace(-10, 10, 100)
    y = np.exp(-x**2/(2*std**2))
    return x, y


#figure of size 6x6 inches
#plt.figure(figsize=(6,6))
plt.figure(figsize=(10,3))

colors=['brown','teal','darkviolet','black']
for s,std in enumerate([1,2, 3, 4]):
    x, y = plot_gaussian(std)
colors=['lightcoral','maroon','mediumturquoise','palegoldenrod','yellowgreen','lightsalmon','tan','grey','peru','lightsteelblue','pink']
for k in range(1,11): #for example for i=2, need to be for x between 1.5 and 2.5
    plt.fill_between(x, 0, plot_gaussian(4)[1], where=(x>=k-0.6) & (x<=k+0.6), color=colors[k-1], alpha=0.4)
    plt.fill_between(x, 0, plot_gaussian(4)[1], where=(x>=-k-0.6) & (x<=-k+0.6), color=colors[k-1], alpha=0.4)

plt.fill_between(x, 0, plot_gaussian(4)[1], where=(x>=-0.6) & (x<=0.6), color='lavender', alpha=0.7)

draw_selection_pairs(3,2,'blue')
draw_selection_pairs(3,0.95,'purple')
draw_selection_pairs(4,1.3,'red')
draw_selection_pairs(4,2.5,'forestgreen')

total_x_axis,total_y_axis=return_x_for_axis(3,2)
total_x_axis+=return_x_for_axis(3,0.95)[0]
total_y_axis+=return_x_for_axis(3,0.95)[1]
total_x_axis+=return_x_for_axis(4,1.3)[0]
total_y_axis+=return_x_for_axis(4,1.3)[1]
total_x_axis+=return_x_for_axis(4,2.5)[0]
total_y_axis+=return_x_for_axis(4,2.5)[1]
total_x_axis=np.array(total_x_axis)
total_y_axis=np.array(total_y_axis)
print("shape of total_x_axis",total_x_axis.shape)
print("shape of total_y_axis",total_y_axis.shape)


plt.ylim(0, 1.0)
plt.xlim(0,4)
plt.xticks(np.arange(0, 10, step=1))
plt.yticks(np.arange(0, 1.1, step=0.2))
plt.xlabel(r'Square $i$', fontsize=15)
plt.ylabel(r'Coefficient $\alpha_i$', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=15)
plt.tight_layout()
plt.grid()
plt.show()
