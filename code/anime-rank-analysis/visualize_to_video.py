import matplotlib.animation as manimation
print(manimation.writers.list())



Writer = manimation.writers['ffmpeg']
dpi=120
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=4000)
myAnimation.save('im.mp4', writer=writer,dpi=dpi)
print("done.")