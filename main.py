import gui
import data
import analysis

if __name__ in {"__main__", "__mp_main__"}:
    data = data.Data()
    analysis = analysis.Analysis(data)

    gui = gui.GUI()
    