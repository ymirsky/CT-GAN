import sys
if len(sys.argv) == 2:
    if sys.argv[1] == '-h':
        print("python "+sys.argv[0]+" <path to dir of scans>")
        print("python "+sys.argv[0]+" <path to dir of scans> <path to save dir>")
        exit(1)

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from procedures.attack_pipeline import *
from utils.equalizer import *
import matplotlib.animation as animation
import time

class GUI(object):
    # If load_path is to a *.dcm or *.mhd file then only this scan is loaded
    # If load_path is to a directory, then all scans are loaded. It is assumed that each scan is in its own subdirectory.
    # save_path is the directory to save the tampered scans (as dicom)
    def __init__(self, load_path, save_path=None):
        # init manipulator
        self.savepath = save_path
        self.filepaths = self._load_paths(load_path)  # load all scans filepaths in path
        self.fileindex = 0
        self.manipulator = scan_manipulator()
        self.manipulator.load_target_scan(self.filepaths[self.fileindex])
        self.hist_state = True
        self.inject_coords = []
        self.remove_coords = []

        # init plot
        self.eq = histEq(self.manipulator.scan)
        self.slices, self.cols, self.rows = self.manipulator.scan.shape
        self.ind = self.slices // 2
        self.pause_start = 0
        self.fig, self.ax = plt.subplots(1, 1, dpi=100)
        self.fig.suptitle('CT-GAN: Malicious Tampering of 3D Medical Imagery using Deep Learning\nTool by Yisroel Mirsky', fontsize=14, fontweight='bold')
        plt.subplots_adjust(bottom=0.2)
        self.ani_direction = 'down'
        self.animation = None
        self.animation_state = True
        self.plot()
        self.ax.set_title(os.path.split(self.filepaths[self.fileindex])[-1]) #filename


        # register click/scroll events
        self.action_state = 'inject' #default state
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

        # register buttons
        axanim = plt.axes([0.1, 0.21, 0.2, 0.075])
        self.banim = Button(axanim, 'Toggle Animation')
        self.banim.on_clicked(self.toggle_animation)

        axinj = plt.axes([0.1, 0.05, 0.1, 0.075])
        axrem = plt.axes([0.21, 0.05, 0.1, 0.075])
        self.binj = Button(axinj, 'Inject')
        self.binj.on_clicked(self.inj_on)
        self.brem = Button(axrem, 'Remove')
        self.brem.on_clicked(self.rem_on)

        axhist = plt.axes([0.35, 0.05, 0.2, 0.075])
        self.bhist = Button(axhist, 'Toggle HistEQ')
        self.bhist.on_clicked(self.hist)

        axprev = plt.axes([0.59, 0.05, 0.1, 0.075])
        axsave = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next)
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev)
        self.bsave = Button(axsave, 'Save')
        self.bsave.on_clicked(self.save)
        self.maximize_window()
        self.update()
        plt.show()

    def _load_paths(self,path):
        filepaths = []
        # load single scan?
        if (path.split('.')[-1] == "dcm") or (path.split('.')[-1] == "mhd"):
            filepaths.append(path)
            return filepaths
        # try load directory of scans...
        files = os.listdir(path)
        for file in files:
            if os.path.isdir(file):
                subdir = os.path.join(path,file)
                subdir_files = os.listdir(subdir)
                if subdir_files[0].split('.')[-1] == "dcm": #folder contains dicom
                    filepaths.append(os.path.join(path,subdir))
                elif (subdir_files[0].split('.')[-1] == "mhd") or (subdir_files[0].split('.')[-1] == "raw"): # MHD
                    filepaths.append(os.path.join(path,subdir,subdir_files[0]))
            elif file.split('.')[-1] == "mhd":
                filepaths.append(os.path.join(path,file))
        return filepaths

    def onclick(self, event):
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))
        if event.xdata is not None:
            coord = np.array([self.ind,event.ydata,event.xdata],dtype=int)
            if coord[1] > 0 and coord[2] > 0:
                self.pause_start = np.Inf #pause while working
                if self.action_state == 'inject':
                    self.ax.set_title("Injecting...")
                    self.im.axes.figure.canvas.draw()
                    self.manipulator.tamper(coord, action='inject', isVox=True)
                    self.inject_coords.append(coord)
                else:
                    self.ax.set_title("Removing...")
                    self.im.axes.figure.canvas.draw()
                    self.manipulator.tamper(coord, action='remove', isVox=True)
                    self.remove_coords.append(coord)
                self.pause_start = time.time() #pause few secs to see result before continue
                self.ax.set_title(os.path.split(self.filepaths[self.fileindex])[-1])  # filename
                self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def toggle_animation(self, event):
        self.animation_state = not self.animation_state
        if self.animation_state:
            self.pause_start = 0
        else:
            self.pause_start = np.Inf

    def inj_on(self, event):
        self.action_state = 'inject'

    def rem_on(self, event):
        self.action_state = 'remove'

    def hist(self, event):
        self.hist_state = not self.hist_state
        self.plot()
        self.update()

    def next(self, event):
        self.fileindex = (self.fileindex + 1) % len(self.filepaths)
        self.loadscan(self.fileindex)

    def prev(self, event):
        self.fileindex = (self.fileindex - 1) % len(self.filepaths)
        self.loadscan(self.fileindex)

    def save(self, event):
        if self.savepath is not None:
            self.ax.set_title("Saving...")
            self.im.axes.figure.canvas.draw()
            uuid = os.path.split(self.filepaths[self.fileindex])[-1][:-4]
            #save scan
            self.manipulator.save_tampered_scan(os.path.join(self.savepath,uuid),output_type='dicom')
            #save coords
            file_exists = False
            if os.path.exists(os.path.join(self.savepath,"tamper_coordinates.csv")):
                file_exists = True
            f = open(os.path.join(self.savepath,"tamper_coordinates.csv"),"a+")
            load_filename = os.path.split(self.filepaths[self.fileindex])[-1]
            if not file_exists:
                f.write("filename, x, y, z, tamper_type\n") #header
            for coord in self.inject_coords:
                f.write(load_filename+", "+str(coord[2])+", "+str(coord[1])+", "+str(coord[0])+", "+"inject\n")
            for coord in self.remove_coords:
                f.write(load_filename+", "+str(coord[2])+", "+str(coord[1])+", "+str(coord[0])+", "+"remove\n")
            f.close()
            self.ax.set_title(load_filename)  # filename
            self.im.axes.figure.canvas.draw()

    def update(self):
        if self.hist_state:
            self.im.set_data(self.eq.equalize(self.manipulator.scan[self.ind,:,:]))
        else:
            self.im.set_data(self.manipulator.scan[self.ind,:,:])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

    def loadscan(self,fileindex):
        #load screen
        self.im.set_data(np.ones((self.cols,self.rows))*-1000)
        self.ax.set_title("Loading...")
        self.im.axes.figure.canvas.draw()
        self.remove_coords.clear()
        self.inject_coords.clear()
        #load scan
        self.manipulator.load_target_scan(self.filepaths[fileindex])
        self.slices, self.cols, self.rows = self.manipulator.scan.shape
        self.ind = self.slices//2
        self.ax.clear()
        self.eq = histEq(self.manipulator.scan)
        self.plot()
        self.ax.set_title(os.path.split(self.filepaths[fileindex])[-1]) #filename
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

    def plot(self):
        self.ax.clear()
        if self.hist_state:
            self.im = self.ax.imshow(self.eq.equalize(self.manipulator.scan[self.ind,:,:]),cmap="bone")#, cmap="bone", vmin=-1000, vmax=1750)
        else:
            self.im = self.ax.imshow(self.manipulator.scan[self.ind,:,:], cmap="bone", vmin=-1000, vmax=1750)
        self.animation = animation.FuncAnimation(self.fig, self.animate, interval=100)

    def animate(self,i):
        if self.animation_state:
            if time.time() - self.pause_start > 1:
                if self.ind == self.slices-1:
                    self.ani_direction = 'up'
                elif self.ind == 0:
                    self.ani_direction = 'down'
                if self.ani_direction == 'up':
                    self.ind-=1
                else:
                    self.ind+=1
                self.update()

    def maximize_window(self):
        try: #'QT4Agg'
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        except:
            try: #'TkAgg'
                mng = plt.get_current_fig_manager()
                mng.window.state('zoomed')
            except:
                try: #'wxAgg'
                    mng = plt.get_current_fig_manager()
                    mng.frame.Maximize(True)
                except:
                    print("Could not maximize window")

if (len(sys.argv) == 1) or (len(sys.argv) > 3):
    loadpath = "data\\healthy_scans"
    savepath = "data\\tampered_scans"
if len(sys.argv) == 2:
    loadpath = sys.argv[1]
    savepath = "data\\tampered_scans"
if len(sys.argv) == 3:
    loadpath = sys.argv[1]
    savepath = sys.argv[2]

gui = GUI(load_path=loadpath,save_path=savepath)
