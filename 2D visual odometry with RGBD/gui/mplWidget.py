# Imports
from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib

# Ensure using PyQt5 backend
#matplotlib.use('QT5Agg')

# Matplotlib canvas class to create figure
class DynamicMplCanvas(Canvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([-1 ,5])
        self.ax.set_ylim([-6.5 , 0.5])
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)
        #self.update_figure()
    def update_figure(self,pred,actual):
        self.ax.cla()
        if len(pred) > 0 and len(actual)>0:
            self.ax.set_xlim([-1 ,5])
            self.ax.set_ylim([-6.5 , 0.5])
            self.ax.plot(pred[:,0],pred[:,1], linestyle = '-' ,color = 'red' , label = 'predicted')
            self.ax.plot(actual[:,0],actual[:,1], linestyle = '-' , color = 'blue' , label = 'actual')
            self.ax.legend()
        self.draw()


# Matplotlib widget
class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.winparent = parent
        self.canvas = DynamicMplCanvas()                  # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)