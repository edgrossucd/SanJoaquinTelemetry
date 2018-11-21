# survey.py
# methods for dealing with an acoustic telemetry survey

import sys, os
import numpy as np
import pandas as pd
from track import Track
#from segment import Segment
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import stompy.grid.unstructured_grid as ugrid
import stompy.plot.cmap as cmap

import pdb

# helper functions
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# global variables and actions
#bathy_cmap = cmap.load_gradient('cyanotype_01.cpt',reverse=True)
bathy_cmap = cmap.load_gradient('cyanotype_01.cpt')
cm_bath = truncate_colormap(bathy_cmap,0.1,0.9)
xylims = [[647100, 647500],[4185600, 4186000]]
#xylims = [[647200, 647400],[4185700, 4185900]]

class Survey(object):

    def __init__(self, 
                 filename = None,
                 grd_file = None):
        self.filename = filename
        self.load_grid(grd_file)
        self.get_tracks()

    def get_tracks(self):
        self.read_survey()
        self.IDs = np.unique(self.dframe['ID'].values)
        self.ntracks = len(self.IDs)
        self.tracks = {}
        for ntag, ID in enumerate(self.IDs):
            dframe_track = self.dframe.loc[self.dframe['ID']==ID]
            # saves dataframe as track.df_track
            self.tracks[ID] = Track(dframe_track, grd=self.grd) 

    def set_directories(self, fig_dir='.', csv_dir='.'):
        self.fig_dir = fig_dir
        self.csv_dir = csv_dir

    def read_survey(self):
        # assume specific csv format for now
        self.dframe = pd.read_csv(self.filename, index_col=0)

    def load_grid(self, grd_file):
        self.grd = ugrid.PtmGrid(grd_file)

    def speed_over_ground(self):
        self.segments = {}
        for nt, key in enumerate(self.tracks.keys()):
            tr = self.tracks[key]
            # save dataframe as segment.df_seg
            self.segments[key] = tr.make_segments(set_depth=True)
            if nt == 0:
                self.df_all = tr.df_seg
            else:
                self.df_all = pd.concat([self.df_all, tr.df_seg])
        self.df_all.to_csv('segments.csv', index=False)
        self.rec_all = self.df_all.to_records()
        # now screen segments? - or screen after hydro speed is estimated?
            
    def plot_all_speed_vectors(self, variable='speed_over_ground'):
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_bathy(ax)
        for nt, key in enumerate(self.tracks.keys()):
            tr = self.tracks[key]
            tr.plot_speed(variable, ax)
        self.format_map(ax)
        fig_name = os.path.join(fig_dir, 'all_vectors.png')
        plt.savefig(fig_name, dpi=800)

    def plot_speed_single_track(self, variable='speed_over_ground', 
                                      color_by='age'):
        for nt, key in enumerate(self.tracks.keys()):
            plt.close()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            self.plot_bathy(ax)
            tr = self.tracks[key]
            color_by = 'age'
            tr.plot_speed(variable, ax, color_by=color_by)
            self.format_map(ax)
            ax.set_title('%s'%(key))
            # save figure
            fig_name = os.path.join(fig_dir, '%s_vectors.png'%(key))
            plt.savefig(fig_name, dpi=800)
        
    def plot_bathy(self, ax):
        self.s_coll = self.grd.plot_cells(values=-self.grd.cells['depth'],
                                          cmap=cm_bath,edgecolor='none',
                                          ax=ax)
        #self.s_coll.set_clim([-4,12])
        self.s_coll.set_clim([-12,4])

        # could invert ticks here
        #cticks=np.arange(-4,12,4)
        cticks=np.arange(-12,4.1,4)
        #cticklabels = ['%d'%(-ctick) for ctick in cticks]
        self.cbar = plt.colorbar(self.s_coll, 
                                 ticks=cticks,
                                 label='Bed Elevation (m NAVD)')
        #self.cbar.ax.invert_yaxis()

    def plot_detects(self):
        plt.close()
        jet = plt.get_cmap('jet')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ncells = len(self.grd.cells['depth'])
        detects_i = np.zeros(ncells, np.int32)
        for nt, key in enumerate(self.tracks.keys()):
            tr = self.tracks[key]
            detects_i_tr = tr.cell_detects(grd=self.grd)
            detects_i += detects_i_tr
        self.s_coll = self.grd.plot_cells(values=detects_i,
                                          cmap=jet,edgecolor='none', ax=ax)
        self.s_coll.set_clim([0,20])
        self.cbar = plt.colorbar(self.s_coll, label='Detections')
        self.format_map(ax)
        fig_name = os.path.join(fig_dir, 'detects_in_cell.png')
        plt.savefig(fig_name, dpi=800)

    def plot_cell_speed(self):
        plt.close()
        jet = plt.get_cmap('jet')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ncells = len(self.grd.cells['depth'])
        detects_i = np.zeros(ncells, np.int32)
        speed_sum_i = np.zeros(ncells, np.float64)
        for nt, key in enumerate(self.tracks.keys()):
            tr = self.tracks[key]
            detects_i_tr, speed_sum_i_tr = tr.cell_speed()
            detects_i += detects_i_tr
            speed_sum_i += speed_sum_i_tr
        speed_i = np.divide(speed_sum_i, detects_i)
        self.s_coll = self.grd.plot_cells(values=speed_i,
                                          cmap=jet,edgecolor='none', ax=ax)
        self.s_coll.set_clim([0,2])
        self.cbar = plt.colorbar(self.s_coll, label='Speed (m/s)')
        self.format_map(ax)
        fig_name = os.path.join(fig_dir, 'speed_in_cell.png')
        plt.savefig(fig_name, dpi=800)

    def plot_non_detects(self):
        plt.close()
        jet = plt.get_cmap('jet')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ncells = len(self.grd.cells['depth'])
        non_detects_i = np.zeros(ncells, np.int32)
        for nt, key in enumerate(self.tracks.keys()):
            tr = self.tracks[key]
            non_detects_i_tr = tr.nondetects()
            non_detects_i += non_detects_i_tr
        self.s_coll = self.grd.plot_cells(values=non_detects_i,
                                          cmap=jet,edgecolor='none', ax=ax)
        self.s_coll.set_clim([0,50])
        self.cbar = plt.colorbar(self.s_coll, label='Non Detects')
        self.format_map(ax)
        fig_name = os.path.join(fig_dir, 'non_detects.png')
        plt.savefig(fig_name, dpi=800)

    def plot_occupancy(self):
        plt.close()
        jet = plt.get_cmap('jet')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ncells = len(self.grd.cells['depth'])
        occupancy_i = np.zeros(ncells, np.int32)
        for nt, key in enumerate(self.tracks.keys()):
            tr = self.tracks[key]
            i_list = tr.cell_occupancy()
            occupancy_i[i_list] += 1
        self.s_coll = self.grd.plot_cells(values=occupancy_i,
                                          cmap=jet,edgecolor='none', ax=ax)
        self.s_coll.set_clim([0,10])
        self.cbar = plt.colorbar(self.s_coll, label='Occupancy')
        self.format_map(ax)
        fig_name = os.path.join(fig_dir, 'occupancy.png')
        plt.savefig(fig_name, dpi=800)

    def calculate_quality_metrics(self):
        metrics = {}
        for nt, key in enumerate(self.tracks.keys()):
            print key
            tr = self.tracks[key]
            mnames, metrics_tr = tr.track_quality_metrics()
            for nn, mname in enumerate(mnames):
                if nt == 0:
                    metrics[mname] = []
                metrics[mname].append(metrics_tr[nn])

        df_metrics = pd.DataFrame.from_dict(metrics)
        metrics_csv = os.path.join(self.csv_dir, 'metrics.csv')
        df_metrics.to_csv(metrics_csv, index=False)
        self.df_metrics = df_metrics 

    def classify_by_quality(self):
        df = self.df_metrics
        IDs = df['ID'].values
        poor_quality = np.zeros(self.ntracks, np.int32)
        quality_dict = {}
        #dfs = df.sort_values(['frac_detects'])
        exclude_tracks = []
        for nt, key in enumerate(self.tracks.keys()):
            row = np.where(IDs == key)[0]
            frac_detects = df['frac_detects'].values[row]
            vel_sd = df['vel_sd'].values[row]
            quality_dict[key] = '|'
            if frac_detects < 0.1:
                poor_quality[nt] = 1 # 1 = True
                quality_dict[key] += 'nondetects|'
            if vel_sd > 0.6:
                poor_quality[nt] = 1 # 1 = True
                quality_dict[key] += 'vel_sd|'
            if poor_quality[nt] > 0:
                exclude_tracks.append(key)
        print "exclude_tracks: ",exclude_tracks
        self.exclude_tracks = exclude_tracks

        # append to df_metrics

    def format_map(self, ax):
        ax.axis('scaled')
        ax.set_xlim(xylims[0])
        ax.set_ylim(xylims[1])
        show_axis = False
        if show_axis:
            ax.set_xlabel('Easting (m)')
            ax.set_ylabel('Northing (m)')
        else:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.axis('off')
        #ax.autoscale(tight=True)

if __name__ == '__main__':
    filename = r'T:\UCD\Projects\CDFW_Klimley\Observations\telemetry\2018_tracks.csv'
    grd_file = 'FISH_PTM.grd'
    sur = Survey(filename, grd_file) # read survey
    #sur.load_grid(grd_file)
    sur.speed_over_ground()
    fig_dir = r'T:\UCD\Projects\CDFW_Klimley\Analysis\python\Telemetry\figures'
    csv_dir = r'T:\UCD\Projects\CDFW_Klimley\Analysis\python\Telemetry\csv'
    sur.set_directories(fig_dir=fig_dir, csv_dir=csv_dir)
    #sur.plot_occupancy()
    #sur.plot_detects()
    #sur.plot_cell_speed()
    #sur.calculate_quality_metrics()
    #sur.classify_by_quality()
    #sur.plot_non_detects()
    #sur.plot_all_speed_vectors(variable='speed_over_ground')
    sur.plot_speed_single_track(variable='speed_over_ground', color_by='age')
    # do analysis here
