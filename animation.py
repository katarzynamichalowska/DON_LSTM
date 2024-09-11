import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from IPython import display
import numpy as np

def animate_wave(g_u, g_u_pred=None, interval=20, fps=30, save_name=None):
    """
    Animation in time.
    """
    
    fig = plt.figure()
    fig.set_dpi(80)
    if g_u_pred is not None:
        assert(g_u.shape==g_u_pred.shape)
        lims = (np.min([g_u, g_u_pred]), np.max([g_u, g_u_pred]))

    else:
        lims = (g_u.min(), g_u.max())
    ax = plt.axes(ylim=lims)

    line1, = ax.plot(g_u[0])
    if g_u_pred is not None:
        line2, = ax.plot(g_u_pred[0])
        args = (g_u, g_u_pred,)
    else:
        args = (g_u,)

    def _animate(i, u, u_pred=None):
        line1.set_ydata(u[i])
        if u_pred is not None:
            line2.set_ydata(u_pred[i])
            return line1,line2,
        else:
            return line1,
            
    anim = animation.FuncAnimation(
        fig, 
        _animate, 
        fargs = args, 
        interval=interval,
        blit=True,
        frames=range(g_u.shape[0]))

    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    if save_name is not None:
        writervideo = animation.FFMpegWriter(fps=fps)
        anim.save(f'{save_name}.mp4', writer=writervideo)

    plt.close()  

def animate_wave_subplots(g_u, g_u_pred, g_u_pred2, t=np.linspace(0.025,5.025,201), 
                          interval=20, fps=30, save_name=None, modelname1=None, modelname2=None):
    """
    Animation in time with 2 subplots stacked vertically (g_u vs g_u_pred) and (g_u vs g_u_pred2).
    """
    
    # Animation function
    def _animate(i, g_u, g_u_pred, g_u_pred2):
        line_gt1.set_ydata(g_u[i])
        line_pred1.set_ydata(g_u_pred[i])
        line_gt2.set_ydata(g_u[i])
        line_pred2.set_ydata(g_u_pred2[i])
        time.set_text(f"t={t[i]:.3f}")

        return [line_gt1, line_pred1, line_gt2, line_pred2]

    # Figure y limits
    lims = (np.min([g_u, g_u_pred, g_u_pred2]), np.max([g_u, g_u_pred, g_u_pred2]))
        
    # Initialize figure
    fig, (ax1, ax2) = plt.subplots(2,1)

    fig.set_dpi(80)
    plt.subplots_adjust(left=0.05,
                        bottom=0.1,
                        right=0.95,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)    
        
    for ax in [ax1, ax2]:
        ax.set_ylim(lims)
        
    ax1.set_xticks([])

    line_gt1, = ax1.plot(g_u[0])
    line_pred1, = ax1.plot(g_u_pred[0])    
    line_gt2, = ax2.plot(g_u[0])
    line_pred2, = ax2.plot(g_u_pred2[0])

    args = (g_u, g_u_pred, g_u_pred2,)
    
    time = ax1.text(0.065,1.065, "t=0.00", bbox={'facecolor':'w', 'edgecolor':'w', 'alpha':0.5, 'pad':0},
                transform=ax1.transAxes, ha="center", fontsize=12)

    ax1.text(0.5,1.065, modelname1, bbox={'facecolor':'w', 'edgecolor':'w', 'alpha':0.5, 'pad':0},
                transform=ax1.transAxes, ha="center", fontsize=12)
    ax2.text(0.5,1.065, modelname2, bbox={'facecolor':'w', 'edgecolor':'w', 'alpha':0.5, 'pad':0},
                transform=ax2.transAxes, ha="center", fontsize=12)
            
    anim = animation.FuncAnimation(
        fig, 
        _animate,       # animation function
        fargs = args,   # arguments to the animation function
        interval=interval,
        blit=True,
        frames=range(g_u.shape[0]))

    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    if save_name is not None:
        writervideo = animation.FFMpegWriter(fps=fps)
        anim.save(f'{save_name}.mp4', writer=writervideo)

    plt.close()  

def animate_lines(g_u, interval=20, fps=30, save_name=None):
    """
    Animation in time.
    """
    
    fig = plt.figure(figsize=(13,5))
    fig.set_dpi(80)
    lims = (np.min(g_u), np.max(g_u))
    
    if not isinstance(g_u, list):
        g_u = [g_u]

    ax = plt.axes(ylim=lims)
    
    lines = []
    for g_u_i in g_u:
        line, = ax.plot(g_u_i[0])
        lines.append(line)

    def _animate(i, g_u):
        for g_u_i, l in zip(g_u, lines):
            l.set_ydata(g_u_i[i])
        return lines
            
    anim = animation.FuncAnimation(
        fig, 
        _animate, 
        fargs = (g_u,), 
        interval=interval,
        blit=True,
        frames=range(g_u[0].shape[0]))

    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    if save_name is not None:
        writervideo = animation.FFMpegWriter(fps=fps)
        anim.save(f'{save_name}.mp4', writer=writervideo)

    plt.close()  
