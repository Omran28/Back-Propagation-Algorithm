import tkinter as tk

window = tk.Tk()

window.title('Deep learning model')
window.geometry("700x450")

# No Of hidden layers
noOfLayersTxt = tk.IntVar()
tk.Label(window, text="Enter number of hidden layers ").place(x=15, y=40)
noOfLayers_entry = tk.Entry(window)
noOfLayers_entry.place(x=200, y=40)

# No Of neuron layers
noOfNeuronsTxt = tk.StringVar()
tk.Label(window, text="Enter number of neurons in each layer").place(x=15, y=80)
noOfNeurons_entry = tk.Entry(window)
noOfNeurons_entry.place(x=200, y=80)

# Learning Rate Entry
etaTxt = tk.DoubleVar()
tk.Label(window, text="Enter Learning Rate ").place(x=15, y=120)
eta_entry = tk.Entry(window)
eta_entry.place(x=200, y=120)

# Epochs Entry
epochsTxt = tk.IntVar()
tk.Label(window, text="Enter number of epochs ").place(x=15, y=160)
epochs_entry = tk.Entry(window)
epochs_entry.place(x=200, y=160)

# Bias Checkbox
biasTxt = tk.BooleanVar()

# Activation function
tk.Label(window, text="Choose Activation function ").place(x=15, y=200)
options = ['Sigmoid', 'Hyperbolic']
fun = tk.StringVar()
fun.set('None')
tk.OptionMenu(window, fun, *options).place(x=200, y=200)


def check_changed():
    print("")


tk.Checkbutton(window, text='Check for Bias', command=check_changed, variable=biasTxt, onvalue=True,
               offvalue=False).place(x=15, y=310)


def call_back(e1, e2, e3, e4):
    e1.set(noOfLayers_entry.get())
    e2.set(noOfNeurons_entry.get())
    e3.set(eta_entry.get())
    e4.set(epochs_entry.get())

    window.destroy()


tk.Button(window, text='Start', background="red", width=25,
          command=lambda: call_back(noOfLayersTxt, noOfNeuronsTxt, etaTxt, epochsTxt)).place(x=250, y=400)
window.mainloop()
