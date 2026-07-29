'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import matplotlib
matplotlib.use('Agg')
from uis.ui_vis import startUi

if __name__ == '__main__':
    path_cfg = './configs/cfg_GUI_TOOL.yml'
    startUi(path_cfg)
